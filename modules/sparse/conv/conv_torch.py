import torch
import torch.nn as nn
import torch.nn.functional as F
from .. import SparseTensor


def _pack_coords(coords: torch.Tensor, stride_bits: int = 12) -> torch.Tensor:
    """Pack [N, 4] int coords (b, x, y, z) into unique int64 keys."""
    c = coords.to(torch.int64)
    return (
        (c[:, 0] << (stride_bits * 3))
        | (c[:, 1] << (stride_bits * 2))
        | (c[:, 2] << stride_bits)
        | c[:, 3]
    )


class _PureSubMConv3d(nn.Module):
    """Pure-PyTorch SubMConv3d matching spconv.SubMConv3d parameter layout (out_c, kD, kH, kW, in_c)."""

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, bias=True, indice_key=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size, kernel_size)
        else:
            self.kernel_size = tuple(kernel_size)
        if isinstance(dilation, int):
            self.dilation = (dilation, dilation, dilation)
        else:
            self.dilation = tuple(dilation)
        self.indice_key = indice_key

        kD, kH, kW = self.kernel_size
        self.weight = nn.Parameter(torch.empty(out_channels, kD, kH, kW, in_channels))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)
        nn.init.kaiming_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, feats: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        kD, kH, kW = self.kernel_size
        if kD == 1 and kH == 1 and kW == 1:
            w = self.weight.reshape(self.out_channels, self.in_channels)
            return F.linear(feats, w, self.bias)

        N = coords.shape[0]
        device = feats.device
        out = feats.new_zeros((N, self.out_channels))
        if self.bias is not None:
            out = out + self.bias

        if N == 0:
            return out

        # Sort keys once for O(N log N) vectorized neighbor lookup
        keys = _pack_coords(coords)
        sorted_keys, sort_order = torch.sort(keys)

        half_d = kD // 2
        half_h = kH // 2
        half_w = kW // 2
        d0, d1, d2 = self.dilation

        coords_i64 = coords.to(torch.int64)
        for kd in range(kD):
            od = (kd - half_d) * d0
            for kh in range(kH):
                oh = (kh - half_h) * d1
                for kw in range(kW):
                    ow = (kw - half_w) * d2
                    w_slice = self.weight[:, kd, kh, kw, :]  # (out_c, in_c)
                    if od == 0 and oh == 0 and ow == 0:
                        out = out + F.linear(feats, w_slice)
                        continue

                    nbr = coords_i64.clone()
                    nbr[:, 1] += od
                    nbr[:, 2] += oh
                    nbr[:, 3] += ow
                    valid_bounds = (
                        (nbr[:, 1] >= 0)
                        & (nbr[:, 2] >= 0)
                        & (nbr[:, 3] >= 0)
                        & (nbr[:, 1] < 4096)
                        & (nbr[:, 2] < 4096)
                        & (nbr[:, 3] < 4096)
                    )
                    nbr_keys = _pack_coords(nbr.clamp(min=0, max=4095))
                    pos = torch.searchsorted(sorted_keys, nbr_keys)
                    pos_clamped = pos.clamp(max=N - 1)
                    hit = valid_bounds & (sorted_keys[pos_clamped] == nbr_keys)
                    if hit.any():
                        hit_dst = torch.nonzero(hit, as_tuple=False).squeeze(1)
                        hit_src = sort_order[pos_clamped[hit_dst]]
                        out[hit_dst] = out[hit_dst] + F.linear(feats[hit_src], w_slice)
        return out


class SparseConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, padding=None, bias=True, indice_key=None):
        super().__init__()
        self.stride = tuple(stride) if isinstance(stride, (list, tuple)) else (stride, stride, stride)
        self.padding = padding
        if any(s != 1 for s in self.stride) or (padding is not None):
            raise NotImplementedError("Pure-PyTorch SparseConv3d fallback only supports SubMConv3d (stride=1, padding=None).")
        self.conv = _PureSubMConv3d(
            in_channels, out_channels, kernel_size, dilation=dilation, bias=bias, indice_key=indice_key
        )

    def forward(self, x: SparseTensor) -> SparseTensor:
        dtype_ = x.feats.dtype
        feats = x.feats.to(torch.float32)
        out_feats = self.conv(feats, x.coords).to(dtype_)
        return x.replace(out_feats)


class SparseInverseConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, bias=True, indice_key=None):
        super().__init__()
        self.stride = tuple(stride) if isinstance(stride, (list, tuple)) else (stride, stride, stride)
        self.conv = _PureSubMConv3d(
            in_channels, out_channels, kernel_size, dilation=dilation, bias=bias, indice_key=indice_key
        )

    def forward(self, x: SparseTensor) -> SparseTensor:
        return x.replace(self.conv(x.feats, x.coords))
