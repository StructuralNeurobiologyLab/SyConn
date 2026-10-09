"""Exact, batch-independent submanifold 3D convolution for inference (drop-in for spconv.SubMConv3d inside PTv3).

spconv (2.3.8) gives wrong results in some regimes (Native algorithm on small clouds with larger grid
coordinates; MaskImplicitGemm for 5^3 kernels), so at inference the two submanifold convolutions of PTv3 (stem 5^3,
per-block CPE 3^3) are evaluated exactly: out[i] = b + sum_o W_o^T f[j(i,o)] over all kernel offsets o for which the
neighbour voxel j exists (padding is irrelevant for submanifold convs). The neighbour table (N, kv) is built once per
stage and kernel size by a sorted-key lookup and cached in the point dict (shared by all blocks of the stage, like
spconv's indice_key), the conv itself is one gather + one GEMM.
"""
import itertools
import torch
import torch.nn as nn
import spconv.pytorch as spconv

from .modules import PointModule, PointSequential

_S = 1024  # key stride per axis (grid coords < 512, offsets |d| <= 2 -> no wrap-around)


class GatherSubMConv3d(PointModule):
    def __init__(self, conv: spconv.SubMConv3d):
        super().__init__()
        self.kernel = int(conv.kernel_size[0])
        r = self.kernel // 2
        w = conv.weight.detach()  # (out, kx, ky, kz, in)
        self.register_buffer("weight", w.reshape(w.shape[0], -1, w.shape[-1]).permute(1, 2, 0).reshape(-1, w.shape[0]).contiguous())  # (kv*in, out)
        self.bias = None if conv.bias is None else nn.Parameter(conv.bias.detach().clone(), requires_grad=False)
        offs = torch.tensor(list(itertools.product(range(-r, r + 1), repeat=3)), dtype=torch.long)  # (kv, 3)
        self.register_buffer("key_offs", (offs[:, 0] * _S + offs[:, 1]) * _S + offs[:, 2])
        self.kv = int(offs.shape[0])

    @torch.no_grad()
    def neighbours(self, point):
        ck = f"subm_nbr_k{self.kernel}"
        if ck in point.keys() and point[ck].shape[0] == point.feat.shape[0]:
            return point[ck]
        g = point.grid_coord.long()
        key = ((point.batch.long() * _S + g[:, 0] + 2) * _S + g[:, 1] + 2) * _S + g[:, 2] + 2
        skey, sidx = torch.sort(key)
        nk = (key[:, None] + self.key_offs[None, :]).reshape(-1)
        pos = torch.searchsorted(skey, nk).clamp_(max=skey.shape[0] - 1)
        hit = skey[pos] == nk
        nbr = torch.full_like(nk, -1)
        nbr[hit] = sidx[pos[hit]]
        nbr = nbr.view(-1, self.kv)
        point[ck] = nbr
        return nbr

    def forward(self, point):
        nbr = self.neighbours(point)
        feat = point.feat
        dt = torch.bfloat16 if torch.is_autocast_enabled() else feat.dtype
        featz = torch.cat([feat.to(dt), feat.new_zeros(1, feat.shape[1], dtype=dt)])  # zero row for missing neighbours
        idx = torch.where(nbr >= 0, nbr, feat.shape[0])
        x = featz[idx]  # (N, kv, in), gathered in the compute dtype
        out = x.reshape(x.shape[0], -1) @ self.weight.to(dt)
        if self.bias is not None:
            out = out + self.bias.to(out.dtype)
        point.feat = out
        if "sparse_conv_feat" in point.keys():
            point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(out)
        return point


def replace_subm_convs(model: nn.Module) -> int:
    n = 0
    for mod in list(model.modules()):
        if isinstance(mod, PointSequential):
            for name, child in list(mod._modules.items()):
                if isinstance(child, spconv.SubMConv3d):
                    mod._modules[name] = GatherSubMConv3d(child).to(child.weight.device)
                    n += 1
    return n
