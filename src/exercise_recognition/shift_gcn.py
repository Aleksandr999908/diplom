"""
Shift-GCN (адаптация kchengiva/Shift-GCN) для V=17 и чистого PyTorch.
Временной блок Shift заменён на TCN (Conv2d по времени).

v2: шире каналы (80–160–320), дополнительный графово-временной блок, пулинг с вниманием по (T×V).
Старые веса без поля arch в checkpoint несовместимы — нужно переобучение.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn


def conv_init(m: nn.Conv2d) -> None:
    nn.init.kaiming_normal_(m.weight, mode="fan_out")
    if m.bias is not None:
        nn.init.constant_(m.bias, 0)


def bn_init(bn: nn.BatchNorm2d | nn.BatchNorm1d, scale: float) -> None:
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


class tcn(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 9, stride: int = 1):
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class Shift_tcn(nn.Module):
    """Темпоральная свёртка вместо оригинального CUDA Shift (тот же receptive field по идее)."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 9, stride: int = 1):
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        bn_init(self.bn, 1)
        bn_init(self.bn2, 1)
        conv_init(self.conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn(x)
        x = self.relu(self.conv(x))
        x = self.bn2(x)
        return x


class Shift_gcn(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_point: int):
        super().__init__()
        self.num_point = num_point
        self.in_channels = in_channels
        self.out_channels = out_channels
        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.down = nn.Identity()

        self.Linear_weight = nn.Parameter(
            torch.zeros(in_channels, out_channels, dtype=torch.float32)
        )
        nn.init.normal_(self.Linear_weight, 0, math.sqrt(1.0 / out_channels))
        self.Linear_bias = nn.Parameter(torch.zeros(1, 1, out_channels, dtype=torch.float32))

        self.Feature_Mask = nn.Parameter(torch.ones(1, num_point, in_channels, dtype=torch.float32))
        nn.init.constant_(self.Feature_Mask, 0)

        self.bn = nn.BatchNorm1d(num_point * out_channels)
        self.relu = nn.ReLU(inplace=True)

        index_array = np.empty(num_point * in_channels, dtype=np.int64)
        for i in range(num_point):
            for j in range(in_channels):
                index_array[i * in_channels + j] = (i * in_channels + j + j * in_channels) % (
                    in_channels * num_point
                )
        self.register_buffer("shift_in", torch.from_numpy(index_array), persistent=False)

        index_array2 = np.empty(num_point * out_channels, dtype=np.int64)
        for i in range(num_point):
            for j in range(out_channels):
                index_array2[i * out_channels + j] = (i * out_channels + j - j * out_channels) % (
                    out_channels * num_point
                )
        self.register_buffer("shift_out", torch.from_numpy(index_array2), persistent=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)

    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        n, c, t, v = x0.size()
        x = x0.permute(0, 2, 3, 1).contiguous()
        x = x.view(n * t, v * c)
        x = x[:, self.shift_in]
        x = x.view(n * t, v, c)
        x = x * (torch.tanh(self.Feature_Mask) + 1)
        x = torch.einsum("nvc,cd->nvd", x, self.Linear_weight).contiguous()
        x = x + self.Linear_bias
        x = x.view(n * t, -1)
        x = x[:, self.shift_out]
        x = self.bn(x)
        x = x.view(n, t, v, self.out_channels).permute(0, 3, 1, 2)
        x = x + self.down(x0)
        x = self.relu(x)
        return x


class TCN_GCN_unit(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_point: int,
        stride: int = 1,
        residual: bool = True,
    ):
        super().__init__()
        self.gcn1 = Shift_gcn(in_channels, out_channels, num_point)
        self.tcn1 = Shift_tcn(out_channels, out_channels, stride=stride)
        self.relu = nn.ReLU(inplace=True)
        if not residual:
            self.residual = lambda x: 0  # type: ignore[assignment, misc]
        elif in_channels == out_channels and stride == 1:
            self.residual = lambda x: x
        else:
            self.residual = tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.tcn1(self.gcn1(x)) + self.residual(x))


class TemporalAttentionBlock(nn.Module):
    """Лёгкий temporal self-attention по оси T для каждого сустава (после GCN-стека)."""

    def __init__(self, d_model: int, nhead: int = 4, dropout: float = 0.1):
        super().__init__()
        ff = min(1536, 4 * d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, tt, v = x.shape
        h = x.permute(0, 3, 2, 1).reshape(n * v, tt, c)
        h = self.enc(h)
        return h.reshape(n, v, tt, c).permute(0, 3, 2, 1)


class AttnSkeletalPool(nn.Module):
    """Взвешенная агрегация по узлам времени и суставов + смесь с средним (стабильность)."""

    def __init__(self, channels: int, mean_mix: float = 0.45):
        super().__init__()
        self.mean_mix = float(mean_mix)
        hid = max(channels // 4, 48)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hid),
            nn.ReLU(inplace=True),
            nn.Linear(hid, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: N, C, T, V
        n, c, t, v = x.shape
        h = x.permute(0, 2, 3, 1).reshape(n, t * v, c)
        logits = self.mlp(h).squeeze(-1)
        w = torch.softmax(logits, dim=1)
        attn = (h * w.unsqueeze(-1)).sum(dim=1)
        mean = h.mean(dim=1)
        m = self.mean_mix
        return (1.0 - m) * attn + m * mean


class ShiftGCNClassifier(nn.Module):
    """Вход: (N, C, T, V, M); C — координаты + движение (например 6), M=1."""

    def __init__(
        self,
        num_class: int,
        num_point: int = 17,
        num_person: int = 1,
        in_channels: int = 6,
        base_ch: int = 80,
        mid_ch: int = 160,
        out_ch: int = 320,
        dropout: float = 0.32,
        head_dim: int = 448,
        attn_pool: bool = True,
        extra_block: bool = True,
        temporal_attn: bool = True,
        temporal_attn_heads: int = 4,
    ):
        super().__init__()
        self.num_class = num_class
        self.num_point = num_point
        self.num_person = num_person
        self.in_channels = in_channels
        self.base_ch = base_ch
        self.mid_ch = mid_ch
        self.out_ch = out_ch
        self.dropout_cfg = dropout
        self.head_dim_cfg = head_dim
        self.use_attn_pool = attn_pool
        self.extra_block = extra_block
        self.use_temporal_attn = temporal_attn
        self._temporal_attn_heads = int(temporal_attn_heads)

        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        self.l1 = TCN_GCN_unit(in_channels, base_ch, num_point, residual=False)
        self.l2 = TCN_GCN_unit(base_ch, base_ch, num_point)
        self.l3 = TCN_GCN_unit(base_ch, base_ch, num_point)
        self.l4 = TCN_GCN_unit(base_ch, base_ch, num_point)
        self.l5 = TCN_GCN_unit(base_ch, mid_ch, num_point, stride=2)
        self.l6 = TCN_GCN_unit(mid_ch, mid_ch, num_point)
        self.l7 = TCN_GCN_unit(mid_ch, mid_ch, num_point)
        self.l8 = TCN_GCN_unit(mid_ch, out_ch, num_point, stride=2)
        self.l9 = TCN_GCN_unit(out_ch, out_ch, num_point)
        self.l10 = TCN_GCN_unit(out_ch, out_ch, num_point)
        self.l11: TCN_GCN_unit | None
        if extra_block:
            self.l11 = TCN_GCN_unit(out_ch, out_ch, num_point)
        else:
            self.l11 = None

        self.temporal_attn_mod: TemporalAttentionBlock | None
        if temporal_attn:
            self.temporal_attn_mod = TemporalAttentionBlock(
                out_ch, nhead=self._temporal_attn_heads, dropout=min(0.15, dropout)
            )
        else:
            self.temporal_attn_mod = None

        self.attn_pool_mod: AttnSkeletalPool | None
        if attn_pool:
            self.attn_pool_mod = AttnSkeletalPool(out_ch)
        else:
            self.attn_pool_mod = None

        d = dropout
        self.head = nn.Sequential(
            nn.Dropout(d),
            nn.Linear(out_ch, head_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(d * 0.55),
            nn.Linear(head_dim, num_class),
        )
        nn.init.kaiming_normal_(self.head[1].weight, mode="fan_out")
        nn.init.constant_(self.head[1].bias, 0)
        nn.init.normal_(self.head[4].weight, 0, math.sqrt(2.0 / num_class))
        nn.init.constant_(self.head[4].bias, 0)
        bn_init(self.data_bn, 1)

    def arch_config(self) -> dict[str, Any]:
        return {
            "base_ch": self.base_ch,
            "mid_ch": self.mid_ch,
            "out_ch": self.out_ch,
            "dropout": self.dropout_cfg,
            "head_dim": self.head_dim_cfg,
            "attn_pool": self.use_attn_pool,
            "extra_block": self.extra_block,
            "temporal_attn": self.use_temporal_attn,
            "temporal_attn_heads": self._temporal_attn_heads,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x)
        if self.l11 is not None:
            x = self.l11(x)

        if self.temporal_attn_mod is not None:
            x = x + 0.5 * self.temporal_attn_mod(x)

        c_new = x.size(1)
        tt, vv = x.size(2), x.size(3)
        x = x.view(N, M, c_new, tt, vv)
        # mean по людям; при M=1 эквивалентно squeeze — без ветвления по int (ONNX-трассировка)
        x = x.mean(dim=1)

        if self.attn_pool_mod is not None:
            x = self.attn_pool_mod(x)
        else:
            x = x.mean(dim=(2, 3))

        return self.head(x)
