"""PTv3 cell type inference over SSVs of a SyConn working directory (one GPU per job).

Per SSV: `redundancy` contexts as in the existing cell type pipeline (see ctinfer.py), identical contexts computed once,
logits expanded back to (redundancy, C); prediction = majority vote of the argmax over contexts,
certainty = 1 - H(mean softmax) / ln C (`certainty_estimate` of prediction_pts).

Loading: DataLoader workers each take a chunk of SSVs, prefetch the two pickles per SSV with a thread pool
(the file latency of nvmescratch is the cost, not CPU), build the contexts and hand flat point arrays to the main
process, which packs contexts of many SSVs into GPU batches with a point budget.

Output: <out>.npz with ids, cls, certainty, n_unique, logits (n, redundancy, C) float16 (unique logits in float32 would be
exact; float16 halves the file size), plus timing info.
"""
import argparse, os, sys, time, threading, queue
import numpy as np
os.environ.setdefault('OMP_NUM_THREADS', '1')

parser = argparse.ArgumentParser()
parser.add_argument('--ids', required=True, help='npy with SSV ids to process')
parser.add_argument('--out', required=True)
parser.add_argument("--ckpt", required=True, help="PTv3 cell type model state dict")
parser.add_argument('--width', type=int, default=32)
parser.add_argument('--num_classes', type=int, default=17)
parser.add_argument('--redundancy', type=int, default=20)
parser.add_argument('--ctx', type=float, default=20000)
parser.add_argument('--npoints', type=int, default=50000)
parser.add_argument('--workers', type=int, default=12)
parser.add_argument('--chunk', type=int, default=32, help='SSVs per loader item')
parser.add_argument('--io_threads', type=int, default=4)
parser.add_argument('--batch_pts', type=int, default=400000, help='point budget per GPU batch')
parser.add_argument('--batch_clouds', type=int, default=4000, help='max contexts per GPU batch')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--limit', type=int, default=0)
parser.add_argument('--temperature', type=float, default=1.0, help='logit temperature for the certainty (calibrated on held-out GT contexts); the merge script can recompute the certainty with a different temperature')
args = parser.parse_args()

import torch
from concurrent.futures import ThreadPoolExecutor
from syconn.cnn.ptv3 import ctinfer
from syconn.cnn.ptv3.celltype import PTv3CellType, set_inference_mode

ids = np.load(args.ids).astype(np.uint64)
if args.limit:
    ids = ids[:args.limit]
# process in random order: the id lists are size-sorted, and a chunk of only giant cells (hundreds of MB of mesh each)
# per worker exhausts the memory; a random mix also keeps the GPU load even. Outputs are indexed by id, so order is free.
ids = ids[np.random.default_rng(args.seed).permutation(len(ids))]
R, C = args.redundancy, args.num_classes


class Chunks(torch.utils.data.Dataset):
    def __init__(self, ids):
        self.chunks = [ids[i:i + args.chunk] for i in range(0, len(ids), args.chunk)]

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, i):
        sids = [int(s) for s in self.chunks[i]]
        rng = np.random.default_rng(args.seed * 1_000_003 + i)
        out = []
        with ThreadPoolExecutor(args.io_threads) as ex:
            futs = {}
            order = list(sids); nxt = 0
            # bounded prefetch: at most 2 x io_threads raw pickles in flight (giant cells have meshes of hundreds of MB)
            while nxt < len(order) and len(futs) < 2 * args.io_threads:
                futs[order[nxt]] = ex.submit(ctinfer.read_raw, order[nxt]); nxt += 1
            for sid in order:
                sk, mesh = futs.pop(sid).result()
                if nxt < len(order):
                    futs[order[nxt]] = ex.submit(ctinfer.read_raw, order[nxt]); nxt += 1
                nodes, edges, verts, labels = ctinfer.build_cloud(sk, mesh)
                del sk, mesh
                ctxs, inv, src = ctinfer.contexts(nodes, edges, verts, labels, redundancy=R, ctx_size=args.ctx,
                                                  npoints=args.npoints, rng=rng)
                out.append((sid, ctxs, inv.astype(np.int64)))
        return out


def collate(x):
    return x[0]


loader = torch.utils.data.DataLoader(Chunks(ids), batch_size=1, num_workers=args.workers, collate_fn=collate,
                                     prefetch_factor=2, persistent_workers=False)
dev = 'cuda'
w = args.width
model = PTv3CellType(in_channels=5, num_classes=C, grid_size=0.05, enc_channels=(w, 2 * w, 4 * w, 8 * w, 16 * w),
                     enc_num_head=(max(1, w // 16), max(2, w // 8), max(4, w // 4), max(8, w // 2), max(16, w)),
                     drop_path=0.0).to(dev)
model.load_state_dict(torch.load(args.ckpt, map_location=dev, weights_only=True))
set_inference_mode(model)

n = len(ids)
res_cls = np.full(n, -1, np.int16); res_cert = np.full(n, np.nan, np.float32); res_nuniq = np.zeros(n, np.int16)
res_logits = np.zeros((n, R, C), np.float16)
pos = {int(s): i for i, s in enumerate(ids)}
eye = torch.eye(5, device=dev)

# pending contexts: list of (ssv_row, ctx_idx, coords, labels); flushed to the GPU by point budget
pend, pend_pts = [], 0
per_ssv = {}  # row -> [inv, n_unique, logits list]
t_gpu = 0.0; n_ctx = 0; n_pts = 0


def finish(row):
    inv, nu, lg = per_ssv.pop(row)
    lg = np.stack(lg)  # (nu, C) float32 logits
    full = lg[inv]  # (R, C)
    cls = np.bincount(full.argmax(1), minlength=C).argmax()
    p = torch.softmax(torch.from_numpy(full / args.temperature), -1).numpy()  # temperature-scaled certainty
    pm = p.mean(0); ent = -(pm * np.log(np.clip(pm, 1e-30, None))).sum()
    res_cls[row] = cls; res_cert[row] = 1 - ent / np.log(C); res_nuniq[row] = nu; res_logits[row] = full


def flush():
    global pend, pend_pts, t_gpu, n_ctx, n_pts
    if not pend:
        return
    torch.cuda.synchronize(); t = time.time()
    coords = torch.from_numpy(np.concatenate([c for _, _, c, _ in pend])).to(dev, non_blocking=True)
    labs = torch.from_numpy(np.concatenate([l for _, _, _, l in pend]).astype(np.int64)).to(dev, non_blocking=True)
    lens = torch.tensor([len(c) for _, _, c, _ in pend], device=dev)
    batch = torch.arange(len(pend), device=dev).repeat_interleave(lens)
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        out = model.forward_flat(eye[labs], coords, batch, len(pend)).float().cpu().numpy()
    torch.cuda.synchronize(); t_gpu += time.time() - t; n_ctx += len(pend); n_pts += int(lens.sum())
    for (row, k, _, _), lg in zip(pend, out):
        per_ssv[row][2].append(lg)
        if len(per_ssv[row][2]) == per_ssv[row][1]:
            finish(row)
    pend, pend_pts = [], 0


t0 = time.time(); done = 0; t_last = t0
for chunk in loader:
    for sid, ctxs, inv in chunk:
        row = pos[sid]
        if len(ctxs) == 0:  # no vertices / no skeleton: leave as -1 / nan
            res_nuniq[row] = 0; continue
        per_ssv[row] = [inv, len(ctxs), []]
        for k, (c, l) in enumerate(ctxs):
            if pend and (pend_pts + len(c) > args.batch_pts or len(pend) >= args.batch_clouds):
                flush()
            pend.append((row, k, c, l)); pend_pts += len(c)
        done += 1
    if time.time() - t_last > 60:
        el = time.time() - t0
        print(f'{done}/{n} ssvs {el:.0f}s {done / el:.1f} ssv/s; gpu {t_gpu:.0f}s ({t_gpu / el * 100:.0f}%), {n_ctx} ctx, {n_pts / 1e6:.0f} Mpts', flush=True)
        t_last = time.time()
flush()
assert not per_ssv, f'{len(per_ssv)} ssvs unfinished'
el = time.time() - t0
np.savez(args.out, ids=ids, cls=res_cls, certainty=res_cert, n_unique=res_nuniq, logits=res_logits,
         wall=el, gpu=t_gpu, n_ctx=n_ctx, n_pts=n_pts)
print(f'DONE {n} ssvs in {el:.0f}s = {n / el:.1f} ssv/s; gpu {t_gpu:.0f}s ({t_gpu / el * 100:.0f}%); {n_ctx} unique ctx ({n_ctx / n:.2f}/ssv); '
      f'{n_pts / 1e6:.0f} Mpts ({n_pts / max(t_gpu, 1e-9) / 1e6:.2f} Mpts/s gpu); unlabelled {(res_cls < 0).sum()}', flush=True)
