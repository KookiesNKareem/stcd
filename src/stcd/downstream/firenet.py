"""Pretrained FireNet reconstructor (Scheerlinck et al., WACV 2020).

We **reimplement** the FireNet architecture here (rather than importing/executing
the cloned repo) and load *only* the tensor ``state_dict`` from the official
``firenet_1000.pth.tar`` checkpoint with ``weights_only=True`` — safe
deserialisation, no arbitrary-code-execution risk from the pickle.

Architecture (exactly matching the checkpoint's 24 tensors; config
``num_bins=5, base_num_channels=16, recurrent_block_type=convgru,
num_residual_blocks=2, recurrent_blocks={resblock:[0]}, skip_type=no_skip,
final_activation=none``):

    net.head            : ConvLayer(5→16, k3, ReLU) + ConvGRU(16)
    net.resblocks.0     : ResidualBlock(16) + ConvGRU(16)      (RecurrentResidualLayer)
    net.resblocks.1     : ResidualBlock(16)
    net.pred            : Conv 1×1 (16→1, no activation)

Module/attribute names are chosen so ``load_state_dict(strict=True)`` matches the
checkpoint keys verbatim.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn

Tensor = torch.Tensor

_CKPT = os.path.join(os.path.dirname(__file__), "..", "..", "..",
                     "data", "firenet", "firenet_1000.pth.tar")


def firenet_available() -> bool:
    return os.path.isfile(_CKPT)


# --------------------------------------------------------------------------- #
# Layer definitions (match repo submodules.py exactly)
# --------------------------------------------------------------------------- #
class ConvLayer(nn.Module):
    def __init__(self, c_in, c_out, k, stride=1, padding=0, activation="relu"):
        super().__init__()
        self.conv2d = nn.Conv2d(c_in, c_out, k, stride, padding, bias=True)
        self.activation = getattr(torch, activation) if activation else None

    def forward(self, x):
        x = self.conv2d(x)
        return self.activation(x) if self.activation is not None else x


class ConvGRU(nn.Module):
    def __init__(self, input_size, hidden_size, k=3):
        super().__init__()
        pad = k // 2
        self.hidden_size = hidden_size
        self.reset_gate = nn.Conv2d(input_size + hidden_size, hidden_size, k, padding=pad)
        self.update_gate = nn.Conv2d(input_size + hidden_size, hidden_size, k, padding=pad)
        self.out_gate = nn.Conv2d(input_size + hidden_size, hidden_size, k, padding=pad)

    def forward(self, x, prev_state):
        if prev_state is None:
            prev_state = torch.zeros(x.size(0), self.hidden_size, *x.shape[2:],
                                     device=x.device, dtype=x.dtype)
        stacked = torch.cat([x, prev_state], dim=1)
        update = torch.sigmoid(self.update_gate(stacked))
        reset = torch.sigmoid(self.reset_gate(stacked))
        out = torch.tanh(self.out_gate(torch.cat([x, prev_state * reset], dim=1)))
        return prev_state * (1 - update) + out * update


class ResidualBlock(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, stride=1, padding=1, bias=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return self.relu(out + residual)


class RecurrentConvLayer(nn.Module):
    """ConvLayer followed by a ConvGRU (the FireNet head)."""

    def __init__(self, c_in, c_out, k=3):
        super().__init__()
        self.conv = ConvLayer(c_in, c_out, k, stride=1, padding=k // 2, activation="relu")
        self.recurrent_block = ConvGRU(c_out, c_out, k)

    def forward(self, x, prev_state):
        x = self.conv(x)
        state = self.recurrent_block(x, prev_state)
        return state, state


class RecurrentResidualLayer(nn.Module):
    """ResidualBlock followed by a ConvGRU (recurrent resblock 0)."""

    def __init__(self, c_in, c_out, k=3):
        super().__init__()
        self.conv = ResidualBlock(c_in, c_out)
        self.recurrent_block = ConvGRU(c_out, c_out, k)

    def forward(self, x, prev_state):
        x = self.conv(x)
        state = self.recurrent_block(x, prev_state)
        return state, state


class _UNetFire(nn.Module):
    def __init__(self, num_bins=5, base=16):
        super().__init__()
        self.head = RecurrentConvLayer(num_bins, base, k=3)
        self.resblocks = nn.ModuleList([
            RecurrentResidualLayer(base, base, k=3),   # resblocks.0 (recurrent)
            ResidualBlock(base, base),                 # resblocks.1 (plain)
        ])
        self.pred = ConvLayer(base, 1, k=1, padding=0, activation=None)

    def forward(self, x, prev_states):
        if prev_states is None:
            prev_states = [None, None]
        states = []
        x, s = self.head(x, prev_states[0]); states.append(s)
        x, s = self.resblocks[0](x, prev_states[1]); states.append(s)
        x = self.resblocks[1](x)
        return self.pred(x), states


class FireNet(nn.Module):
    def __init__(self, num_bins=5, base=16):
        super().__init__()
        self.net = _UNetFire(num_bins, base)

    def forward(self, x, prev_states):
        return self.net(x, prev_states)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_firenet(device: str = "cpu", ckpt: str | None = None):
    """Build FireNet and load the pretrained tensor weights (safe load)."""
    ckpt = ckpt or _CKPT
    raw = torch.load(ckpt, map_location="cpu", weights_only=True)
    state = raw["state_dict"] if isinstance(raw, dict) and "state_dict" in raw else raw
    num_bins = 5
    if isinstance(raw, dict) and "config" in raw:
        try:
            num_bins = int(raw["config"]["model"]["num_bins"])
        except Exception:
            pass
    model = FireNet(num_bins=num_bins, base=16)
    model.load_state_dict(state, strict=True)
    model.eval().to(device)
    return FireNetReconstructor(model, num_bins=num_bins, device=device)


class FireNetReconstructor:
    """Recurrent wrapper: feed normalised voxel grids, get intensity frames."""

    def __init__(self, model: FireNet, num_bins: int = 5, device: str = "cpu"):
        self.model = model
        self.num_bins = num_bins
        self.device = device
        self.states = None

    def reset(self) -> None:
        self.states = None

    @torch.no_grad()
    def __call__(self, voxel: Tensor) -> Tensor:  # [1, num_bins, H, W]
        img, self.states = self.model(voxel.to(self.device), self.states)
        return img
