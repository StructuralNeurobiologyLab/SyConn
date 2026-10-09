"""Vendored Point Transformer V3 (Pointcept, commit 1342eda) with minimal stubs; see model.py and LICENSE."""
import spconv.pytorch as _spconv

# spconv's implicit-GEMM auto-tuner can fail ("can't find suitable algorithm") on some batches;
# pin submanifold convolutions to the native algorithm.
if not getattr(_spconv, "_ptv3_native_patch", False):
    _orig = _spconv.SubMConv3d

    class SubMConv3d(_orig):  # noqa: D101
        def __init__(self, *a, **kw):
            kw.setdefault("algo", _spconv.ConvAlgo.Native)
            super().__init__(*a, **kw)

    _spconv.SubMConv3d = SubMConv3d
    _spconv._ptv3_native_patch = True

from .model import PointTransformerV3  # noqa: E402
from .structure import Point  # noqa: E402
