"""Merge the per-job outputs of exec_celltype_ptv3_inference.py into arrays aligned with <wd>/ssv_0/id.npy (the array format of the existing cell type prediction):
celltype_pts_<key>.npy (class per SSV, -1 = not processed), _certainty, _nunique (unique contexts per SSV) and _probas
(logits, (n, redundancy, C) float16). The certainty is recomputed from the stored logits with the calibrated temperature
(1 - H(mean softmax) / ln C, see Schubert et al. 2022).

    python exec_celltype_ptv3_merge.py --wd <working dir> --parts "<glob of part npz>" --out <array dir> \
        --key celltype_pts_v8 --temperature <npy with T or float>
"""
import argparse, glob, os, time
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--wd", required=True, help="SyConn working directory (ssv_0/id.npy gives the SSV order)")
parser.add_argument("--parts", required=True, help="glob of the inference outputs (npz with ids, cls, certainty, n_unique, logits)")
parser.add_argument("--out", required=True, help="output directory for the arrays")
parser.add_argument("--key", default="celltype_pts_v8")
parser.add_argument("--temperature", default=None, help="npy file with the calibrated temperature or a float; omitted: keep the inference-time certainty")
args = parser.parse_args()
os.makedirs(args.out, exist_ok=True)
ids = np.load(os.path.join(args.wd, "ssv_0", "id.npy"))
n = len(ids)
order = np.argsort(ids)
sorted_ids = ids[order]
parts = sorted(glob.glob(args.parts))
assert len(parts) > 0, "no parts found"
first = np.load(parts[0])
R, C = first["logits"].shape[1:]
cls = np.full(n, -1, np.int16)
cert = np.full(n, np.nan, np.float32)
nu = np.zeros(n, np.int16)
logits = np.zeros((n, R, C), np.float16)
seen = np.zeros(n, bool)
for f in parts:
    r = np.load(f)
    k = order[np.searchsorted(sorted_ids, r["ids"])]
    assert np.array_equal(ids[k], r["ids"]) and not seen[k].any()
    cls[k] = r["cls"]
    cert[k] = r["certainty"]
    nu[k] = r["n_unique"]
    logits[k] = r["logits"]
    seen[k] = True
    print(f, len(k), flush=True)
assert seen.all(), f"{(~seen).sum()} ids missing"
if args.temperature is not None:
    T = float(np.load(args.temperature)[0]) if args.temperature.endswith(".npy") else float(args.temperature)
    print(f"recomputing the certainty with T = {T:.3f}", flush=True)
    for s in range(0, n, 200000):
        lg = logits[s:s + 200000].astype(np.float32) / T
        lg -= lg.max(-1, keepdims=True)
        pr = np.exp(lg)
        pr /= pr.sum(-1, keepdims=True)
        pm = pr.mean(1)
        c = 1 + (pm * np.log(np.clip(pm, 1e-30, None))).sum(-1) / np.log(C)
        ok = cls[s:s + 200000] >= 0
        cert[s:s + 200000][ok] = c[ok]
np.save(os.path.join(args.out, args.key + ".npy"), cls)
np.save(os.path.join(args.out, args.key + "_certainty.npy"), cert)
np.save(os.path.join(args.out, args.key + "_nunique.npy"), nu)
np.save(os.path.join(args.out, args.key + "_probas.npy"), logits)
print(f"written to {args.out}; unlabelled {(cls < 0).sum()} of {n}")
