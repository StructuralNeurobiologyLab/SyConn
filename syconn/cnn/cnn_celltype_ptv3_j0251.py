"""Cell type classification of j0251 hybrid point clouds with Point Transformer V3 (syconn.cnn.ptv3).

Uses the same data pipeline as ``cnn_celltype_ptcnv_j0251.py`` (CellCloudDataJ0251 on the GT csv / hybrid cloud
cache, same augmentations, stratified k-fold split with k = SYCONN_CT_NFOLDS) and a plain PyTorch loop; the
evaluation reports sample-level accuracy and macro F1 on the held-out fold.
"""
import argparse
import os
import random
import time

import numpy as np
import torch
from torch import nn

parser = argparse.ArgumentParser()
parser.add_argument('--sr', type=str, required=True, help='save root')
parser.add_argument('--na', type=str, default='ptv3_celltype', help='experiment name')
parser.add_argument('--cval', type=int, default=0)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--bs', type=int, default=16)
parser.add_argument('--sp', type=int, default=50000)
parser.add_argument('--ctx', type=int, default=20000)
parser.add_argument('--scale_norm', type=int, default=2000)
parser.add_argument('--grid', type=float, default=0.05, help='voxel size in normalised units (0.05 = 100 nm)')
parser.add_argument('--steps', type=int, default=20000)
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--wd', type=float, default=0.05)
parser.add_argument('--warmup', type=int, default=500)
parser.add_argument('--eval_every', type=int, default=1000)
parser.add_argument('--eval_batches', type=int, default=40, help='validation batches per evaluation')
parser.add_argument('--workers', type=int, default=24)
parser.add_argument('--num_classes', type=int, default=17)
parser.add_argument('--width', type=int, default=32, help='base channels (enc: w,2w,4w,8w,16w)')
parser.add_argument('--depths', type=str, default='2,2,2,6,2')
parser.add_argument('--drop_path', type=float, default=0.1)
parser.add_argument('--label_smoothing', type=float, default=0.0)
parser.add_argument('--frag_p', type=float, default=0.0, help='probability of a synthetic short fragment instead of a context (train)')
parser.add_argument('--eval_frag', type=int, default=160, help='held-out synthetic fragments per evaluation (0: off)')
args = parser.parse_args()

torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
dev = torch.device('cuda')
os.makedirs(args.sr, exist_ok=True)
log_path = os.path.join(args.sr, f'{args.na}_CV{args.cval}_seed{args.seed}.log')


def log(msg):
    line = f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {msg}'
    print(line, flush=True)
    with open(log_path, 'a') as f:
        f.write(line + '\n')


import morphx.processing.clouds as clouds
from syconn.cnn.TrainData import CellCloudDataJ0251
from syconn.handler.prediction_pts import init_hc_cache_gt
from sklearn.metrics import f1_score, accuracy_score, classification_report

t0 = time.time(); init_hc_cache_gt(); log(f'hybrid cloud cache loaded in {time.time() - t0:.0f} s')
train_tf = clouds.Compose([clouds.RandomVariation((-40, 40), distr='normal'), clouds.Center(),
                           clouds.Normalization(args.scale_norm), clouds.RandomRotate(apply_flip=True),
                           clouds.ElasticTransform(res=(40, 40, 40), sigma=6),
                           clouds.RandomScale(distr_scale=0.1, distr='uniform')])
valid_tf = clouds.Compose([clouds.Center(), clouds.Normalization(args.scale_norm)])
ds_kwargs = dict(npoints=args.sp, cv_val=args.cval, cellshape_only=False, use_syntype=True, onehot=True,
                 batch_size=args.bs, ctx_size=args.ctx, map_myelin=False)
valid_ds = CellCloudDataJ0251(transform=valid_tf, train=False, **ds_kwargs)
CT_NAMES = ['DA', 'LMAN', 'HVC', 'MSN', 'STN', 'TAN', 'GPe', 'GPi', 'LTS', 'INT1', 'INT2', 'INT3', 'ASTRO', 'OLIGO',
            'MICRO', 'MIGR', 'FRAG', 'INT4', 'OPC'][:args.num_classes]


def to_dev(batch):
    x = torch.as_tensor(batch['features']).float().to(dev, non_blocking=True)
    p = torch.as_tensor(batch['pts']).float().to(dev, non_blocking=True)
    y = torch.as_tensor(batch['target']).long().reshape(-1).to(dev)
    return x, p, y


@torch.no_grad()
def evaluate(model, n_batches, seed=1234, report=False):
    """Sample-level accuracy / macro F1 on the held-out fold (fixed sampling seed)."""
    model.eval()
    st = np.random.get_state(); np.random.seed(seed); torch.manual_seed(seed)
    ys, ps = [], []
    for i in range(n_batches):
        x, p, y = to_dev(valid_ds[i])
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits = model(x, p)
        ps.append(logits.float().argmax(-1).cpu()); ys.append(y.cpu())
    np.random.set_state(st); torch.manual_seed(args.seed + 99991)
    ys, ps = torch.cat(ys).numpy(), torch.cat(ps).numpy()
    present = np.unique(ys)
    res = dict(acc=accuracy_score(ys, ps), f1=f1_score(ys, ps, labels=present, average='macro'), n=len(ys), n_classes_present=len(present))
    if report:
        log('\n' + classification_report(ys, ps, labels=list(range(args.num_classes)), target_names=CT_NAMES, zero_division=0))
    model.train()
    return res


from syconn.cnn.ptv3.celltype import PTv3CellType

# training: ONE context per cell per item, collated into flat clouds of `bs` different cells
# (the original loader draws all `batch_size` contexts of an item from a single cell -> one label per batch)
train_ds = CellCloudDataJ0251(transform=train_tf, **dict(ds_kwargs, batch_size=1))
train_ds.frag_p = args.frag_p
valid_frag_ds = CellCloudDataJ0251(transform=valid_tf, train=False, **dict(ds_kwargs, batch_size=1))
valid_frag_ds.frag_p = 1.0


@torch.no_grad()
def evaluate_frag(model, n, seed=4321):
    """Sample-level accuracy / macro F1 on synthetic fragments of the held-out cells (flat batches of 16)."""
    model.eval()
    st = np.random.get_state(); np.random.seed(seed); torch.manual_seed(seed)
    ys, ps, npts = [], [], []
    for b in range(0, n, 16):
        items = [valid_frag_ds[i] for i in range(b, min(n, b + 16))]
        d = flat_collate(items)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits = model.forward_flat(d['feat'].to(dev), d['coord'].to(dev), d['batch'].to(dev), len(items))
        ps.append(logits.float().argmax(-1).cpu()); ys.append(d['target'].reshape(-1)); npts += [len(it['pts']) for it in items]
    np.random.set_state(st); torch.manual_seed(args.seed + 99991)
    ys, ps = torch.cat(ys).numpy(), torch.cat(ps).numpy(); present = np.unique(ys)
    model.train()
    return dict(acc=accuracy_score(ys, ps), f1=f1_score(ys, ps, labels=present, average='macro'), n=len(ys), pts=float(np.mean(npts)))


def flat_collate(items):
    feats = [torch.as_tensor(it['features']).reshape(-1, 5).float() for it in items]
    pts = [torch.as_tensor(it['pts']).reshape(-1, 3).float() for it in items]
    batch = torch.cat([torch.full((f.shape[0],), i, dtype=torch.long) for i, f in enumerate(feats)])
    y = torch.cat([torch.as_tensor(it['target']).long().reshape(-1) for it in items])
    return dict(feat=torch.cat(feats), coord=torch.cat(pts), batch=batch, target=y)
def _worker_init(_):
    # one thread per worker: with many workers, per-worker OpenMP/torch thread pools oversubscribe the cores
    import gc as _gc; _gc.disable()
    torch.set_num_threads(1)
    os.environ['OMP_NUM_THREADS'] = '1'


import gc
gc.collect(); gc.freeze()  # keep the GC from touching the 51 GB cache in every forked worker (copy-on-write growth)
loader = torch.utils.data.DataLoader(train_ds, batch_size=args.bs, shuffle=True, num_workers=args.workers,
                                     collate_fn=flat_collate, persistent_workers=False, prefetch_factor=2,
                                     pin_memory=True, worker_init_fn=_worker_init, drop_last=True)
w = args.width
model = PTv3CellType(in_channels=5, num_classes=args.num_classes, grid_size=args.grid,
                     enc_channels=(w, 2 * w, 4 * w, 8 * w, 16 * w), enc_num_head=(max(1, w // 16), max(2, w // 8), max(4, w // 4), max(8, w // 2), max(16, w)),
                     enc_depths=tuple(int(d) for d in args.depths.split(',')), drop_path=args.drop_path).to(dev)
log(f'PTv3 parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M')
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / args.warmup) * 0.5 * (1 + np.cos(np.pi * min(s, args.steps) / args.steps)))
crit = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
step, best = 0, dict(acc=0.0)
t_data = t_gpu = 0.0
t_last = time.time()
losses = []
while step < args.steps:
    for batch in loader:
        t1 = time.time(); t_data += t1 - t_last
        feat = batch['feat'].to(dev, non_blocking=True); coord = batch['coord'].to(dev, non_blocking=True)
        bidx = batch['batch'].to(dev, non_blocking=True); y = batch['target'].to(dev)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss = crit(model.forward_flat(feat, coord, bidx, args.bs), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); step += 1
        losses.append(loss.item())
        torch.cuda.synchronize(); t_last = time.time(); t_gpu += t_last - t1
        if step % 50 == 0:
            log(f'step={step} loss={np.mean(losses[-50:]):.3f} lr={sched.get_last_lr()[0]:.2e} '
                f'data {t_data / step * 1000:.0f} ms/step gpu {t_gpu / step * 1000:.0f} ms/step points {int(feat.shape[0])} labels {len(set(y.tolist()))}')
        if step % args.eval_every == 0 or step == args.steps:
            torch.save(model.state_dict(), os.path.join(args.sr, 'state_dict.pth'))
            try:
                res = evaluate(model, args.eval_batches, report=(step == args.steps))
                log(f'EVAL step={step}: acc={res["acc"]:.4f} macroF1={res["f1"]:.4f} n={res["n"]} classes={res["n_classes_present"]}')
                if args.eval_frag and args.frag_p > 0:
                    rf = evaluate_frag(model, args.eval_frag)
                    log(f'EVALFRAG step={step}: acc={rf["acc"]:.4f} macroF1={rf["f1"]:.4f} n={rf["n"]} pts/frag={rf["pts"]:.0f}')
                if res['acc'] > best['acc']:
                    best = dict(res, step=step); torch.save(model.state_dict(), os.path.join(args.sr, 'state_dict_best.pth'))
            except Exception as ex:
                log(f'EVAL step={step} FAILED: {type(ex).__name__}: {str(ex)[:300]}'); model.train()
            t_last = time.time()
        if step >= args.steps:
            break
log(f'DONE best={best}')
