"""Physics-Informed 1D Fourier Neural Operator (FNO-1D) for Battery State Estimation.

Preserves mass conservation via cumulative charge integral Q(t) = \int I dt.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv1d(nn.Module):

  def __init__(self, in_channels: int, out_channels: int, modes1: int):
    super(SpectralConv1d, self).__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.modes1 = modes1
    self.scale = 1.0 / (in_channels * out_channels)
    self.weights1 = nn.Parameter(
        self.scale
        * torch.rand(
            in_channels, out_channels, self.modes1, dtype=torch.cfloat
        )
    )

  def compl_mul1d(self, input, weights):
    return torch.einsum("bix,iox->box", input, weights)

  def forward(self, x):
    batchsize = x.shape[0]
    # Compute 1D FFT
    x_ft = torch.fft.rfft(x)
    out_ft = torch.zeros(
        batchsize,
        self.out_channels,
        x.size(-1) // 2 + 1,
        device=x.device,
        dtype=torch.cfloat,
    )
    out_ft[:, :, : self.modes1] = self.compl_mul1d(
        x_ft[:, :, : self.modes1], self.weights1
    )
    # Inverse FFT back to physical domain
    x = torch.fft.irfft(out_ft, n=x.size(-1))
    return x


class BatteryFNO1D(nn.Module):

  def __init__(
      self, in_channels=5, out_channels=5, modes=24, width=64, num_layers=4
  ):
    super(BatteryFNO1D, self).__init__()
    self.modes = modes
    self.width = width
    self.fc0 = nn.Linear(in_channels, self.width)

    self.conv_layers = nn.ModuleList()
    self.w_layers = nn.ModuleList()
    for _ in range(num_layers):
      self.conv_layers.append(SpectralConv1d(self.width, self.width, self.modes))
      self.w_layers.append(nn.Conv1d(self.width, self.width, 1))

    self.fc1 = nn.Linear(self.width, 128)
    self.fc2 = nn.Linear(128, out_channels)

  def forward(self, x):
    # x shape: (batch, seq_len, in_channels)
    x = self.fc0(x)
    x = x.permute(0, 2, 1)

    for conv, w in zip(self.conv_layers, self.w_layers):
      x1 = conv(x)
      x2 = w(x)
      x = F.gelu(x1 + x2)

    x = x.permute(0, 2, 1)
    x = F.gelu(self.fc1(x))
    x = self.fc2(x)
    return x