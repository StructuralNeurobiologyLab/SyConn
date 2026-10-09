"""Cell type classifier on Point Transformer V3 (encoder only) for SyConn hybrid point clouds.

Input convention identical to the ConvPoint model: ``feats`` (B, N, C) one-hot object channels and
``pts`` (B, N, 3) coordinates (centred, scale-normalised). Points are voxelised at ``grid_size`` (duplicate
coordinates and near-duplicates merge, features averaged), serialised and encoded by PTv3; the per-cloud
descriptor is the concatenation of mean and max pooled encoder features, followed by an MLP head.
"""
from typing import Tuple

import torch
import torch.nn as nn
import torch_scatter

from .model import PointTransformerV3


class PTv3CellType(nn.Module):
    def __init__(self, in_channels: int = 5, num_classes: int = 17, grid_size: float = 0.05,
                 enc_channels=(32, 64, 128, 256, 512), enc_depths=(2, 2, 2, 6, 2), enc_num_head=(2, 4, 8, 16, 32),
                 enc_patch_size=(48, 48, 48, 48, 48), stride=(2, 2, 2, 2), drop_path=0.1, head_dropout=0.3,
                 enable_flash=False):
        super().__init__()
        self.grid_size = grid_size
        self.backbone = PointTransformerV3(
            in_channels=in_channels, order=("z", "z-trans", "hilbert", "hilbert-trans"), stride=stride,
            enc_depths=enc_depths, enc_channels=enc_channels, enc_num_head=enc_num_head,
            enc_patch_size=enc_patch_size, drop_path=drop_path, enable_flash=enable_flash, enc_mode=True,
            shuffle_orders=True)
        d = enc_channels[-1]
        self.head = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Dropout(head_dropout), nn.Linear(d, num_classes))

    def voxelize(self, feats: torch.Tensor, pts: torch.Tensor) -> dict:
        """(B, N, C), (B, N, 3) -> flat dict with unique voxels per cloud (features averaged, coords averaged)."""
        B, N, C = feats.shape
        batch = torch.arange(B, device=pts.device).repeat_interleave(N)
        return self.voxelize_flat(feats.reshape(-1, C), pts.reshape(-1, 3), batch, B)

    def voxelize_flat(self, feat: torch.Tensor, coord: torch.Tensor, batch: torch.Tensor, B: int) -> dict:
        """Flat clouds: feat (M, C), coord (M, 3), batch (M,) cloud index in [0, B)."""
        gmin = torch_scatter.scatter_min(coord, batch, dim=0, dim_size=B)[0]  # (B, 3) per-cloud origin
        gc = torch.div(coord - gmin[batch], self.grid_size, rounding_mode="floor").long()
        key = torch.stack([batch, gc[:, 0], gc[:, 1], gc[:, 2]], 1)
        _, inv = torch.unique(key, dim=0, return_inverse=True)
        M = int(inv.max().item()) + 1
        feat_v = torch_scatter.scatter_mean(feat, inv, dim=0, dim_size=M)
        coord_v = torch_scatter.scatter_mean(coord, inv, dim=0, dim_size=M)
        batch_v = torch_scatter.scatter_max(batch, inv, dim=0, dim_size=M)[0]
        gc_v = torch_scatter.scatter_max(gc, inv, dim=0, dim_size=M)[0]  # exact integer voxel coords (per-cloud origin)
        # Point expects clouds contiguous in batch order
        order = torch.argsort(batch_v, stable=True)
        out = dict(coord=coord_v[order].contiguous(), feat=feat_v[order].contiguous(),
                   batch=batch_v[order].contiguous(), grid_size=self.grid_size)
        if getattr(self, "per_cloud_grid", False):
            # batch-independent and duplicate-free serialisation grid: PTv3 would otherwise re-derive integer grid
            # coords from the voxel centroids with the batch-wide minimum as origin, which (a) depends on the other
            # clouds in the batch and (b) can map two adjacent voxel centroids onto one cell (duplicate indices,
            # undefined behaviour in spconv).
            out["grid_coord"] = gc_v[order].int().contiguous()
        return out

    def forward(self, feats: torch.Tensor, pts: torch.Tensor) -> torch.Tensor:
        B = feats.shape[0]
        return self._forward_data(self.voxelize(feats.float(), pts.float()), B)

    def forward_flat(self, feat: torch.Tensor, coord: torch.Tensor, batch: torch.Tensor, B: int) -> torch.Tensor:
        return self._forward_data(self.voxelize_flat(feat.float(), coord.float(), batch, B), B)

    def _forward_data(self, data: dict, B: int) -> torch.Tensor:
        point = self.backbone(data)
        b = point.batch
        mean = torch_scatter.scatter_mean(point.feat, b, dim=0, dim_size=B)
        mx = torch_scatter.scatter_max(point.feat, b, dim=0, dim_size=B)[0]
        return self.head(torch.cat([mean, mx], 1))


def set_inference_mode(model: nn.Module, fixed_patch: bool = True, shuffle_orders: bool = False,
                       per_cloud_grid: bool = True, depth: int = 9, exact_subm: bool = True) -> nn.Module:
    """Batch-independent, deterministic inference (every cloud gets the result it would get on its own):
    fixed attention patch size (clouds padded/masked individually instead of the batch-min patch size), no random
    serialisation-order shuffling, per-cloud grid origin and a fixed serialisation depth (2^9 voxels = 25.6 units
    = 51 um at scale 2000 nm / grid 0.05; the training batches had depth 8-9)."""
    from .model import SerializedAttention, SerializedPooling, PointTransformerV3
    for mod in model.modules():
        if isinstance(mod, SerializedAttention):
            mod.fixed_patch = fixed_patch
        if isinstance(mod, (SerializedPooling, PointTransformerV3)):
            mod.shuffle_orders = shuffle_orders
        if isinstance(mod, PointTransformerV3):
            mod.serialization_depth = depth
        if isinstance(mod, PTv3CellType):
            mod.per_cloud_grid = per_cloud_grid
    if exact_subm:
        from .subm import replace_subm_convs
        replace_subm_convs(model)  # exact gather-based submanifold convs (spconv gives wrong results in some regimes)
    return model.eval()
