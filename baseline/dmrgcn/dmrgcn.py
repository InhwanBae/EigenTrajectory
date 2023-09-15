# Baseline model for "Disentangled Multi-Relational Graph Convolutional Network for Pedestrian Trajectory Prediction"
# Source-code referred from DMRGCN at https://github.com/InhwanBae/DMRGCN/blob/main/model/dmrgcn.py

import numpy as np
import torch
import torch.nn as nn
from .normalizer import normalized_laplacian_tilde_matrix
from .gcn import ConvTemporalGraphical
from .dropedge import drop_edge


def clip_adjacency_matrix(A, min=-1e10, max=1e10):
    r"""Returns the clipped Adjacency matrix with min and max values."""

    A_c = A.clamp(min=min, max=max)
    A_c[A_c == min] = 0
    A_c[A_c == max] = 0
    A_c[A_c > 0] = 1
    return A_c


def get_disentangled_adjacency_matrix(A, split=[]):
    r"""Returns the list of clipped Adjacency matrix split by list values."""

    if len(split) == 0:
        return [A]

    split.sort()
    split = split + [1e10]

    A_d = []
    for i in range(len(split) - 1):
        A_d.append(clip_adjacency_matrix(A, min=split[i], max=split[i + 1]))

    return torch.stack(A_d, dim=1)


def k_adjacency(A, k, with_self=False, self_factor=1):
    assert isinstance(A, np.ndarray)
    I = np.eye(len(A), dtype=A.dtype)
    if k == 0:
        return I
    Ak = np.minimum(np.linalg.matrix_power(A + I, k), 1) \
         - np.minimum(np.linalg.matrix_power(A + I, k - 1), 1)
    if with_self:
        Ak += (self_factor * I)
    return Ak


class MultiRelationalGCN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, t_kernel_size=1, t_stride=1, t_padding=0,
                 t_dilation=1, bias=True, relation=2):
        super().__init__()
        self.kernel_size = kernel_size
        self.relation = relation
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels * relation, kernel_size=(t_kernel_size, 1), padding=(t_padding, 0),
                              stride=(t_stride, 1), dilation=(t_dilation, 1), bias=bias)

    def forward(self, x, A):
        # Batch calculation for A matrix
        assert A.size(0) == x.size(0)
        assert A.size(1) == self.relation
        assert A.size(2) == self.kernel_size

        x = self.conv(x)
        x = x.view(x.size(0), self.relation, self.out_channels, x.size(-2), x.size(-1))
        x = torch.einsum('nrtwv,nrctv->nctw', normalized_laplacian_tilde_matrix(drop_edge(A, 0.8, self.training)), x)
        return x.contiguous(), A


class st_dagcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, use_mdn=False, stride=1, dropout=0, residual=True,
                 split=[]):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1

        padding = ((kernel_size[0] - 1) // 2, 0)
        self.use_mdn = use_mdn
        self.split = split

        self.gcns = nn.ModuleList()
        for j in range(len(self.split) + 1):
            self.gcns.append(ConvTemporalGraphical(in_channels, out_channels, kernel_size[1]))

        self.tcn = nn.Sequential(
            nn.PReLU(),
            nn.Conv2d(out_channels, out_channels, (kernel_size[0], 1), (stride, 1), padding),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
            )

        self.prelu = nn.PReLU()

    def forward(self, x, A):

        res = self.residual(x)

        A_ = get_disentangled_adjacency_matrix(A, self.split)

        A_ = torch.split(A_, 1, dim=1)

        for j in range(len(self.split) + 1):
            if j == 0:
                x_a, _ = self.gcns[j](x, A_[j].squeeze(dim=1))
            else:
                x_at, _ = self.gcns[j](x, A_[j].squeeze(dim=1))
                x_a = x_a + x_at

        x = self.tcn(x_a) + res

        if not self.use_mdn:
            x = self.prelu(x)

        return x, A


class st_mrgcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, use_mdn=False, stride=1, dropout=0, residual=True,
                 relation=2):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1

        padding = ((kernel_size[0] - 1) // 2, 0)
        self.use_mdn = use_mdn
        self.relation = relation

        self.gcns = nn.ModuleList()
        for r in range(self.relation):
            self.gcns.append(ConvTemporalGraphical(in_channels, out_channels, kernel_size[1]))

        self.tcn = nn.Sequential(
            nn.PReLU(),
            nn.Conv2d(out_channels, out_channels, (kernel_size[0], 1), (stride, 1), padding),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
            )

        self.prelu = nn.PReLU()

    def forward(self, x, A):
        res = self.residual(x)
        A_ = torch.split(A[:, 0:2], 1, dim=1)

        for r in range(self.relation):
            if r == 0:
                x_r, _ = self.gcns[r](x, A_[r].squeeze(dim=1))
            else:
                x_rt, _ = self.gcns[r](x, A_[r].squeeze(dim=1))
                x_r = x_r + x_rt

        x = self.tcn(x_r) + res

        if not self.use_mdn:
            x = self.prelu(x)

        return x, A


class st_dmrgcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, use_mdn=False, stride=1, dropout=0, residual=True,
                 split = [], relation=2):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1

        padding = ((kernel_size[0] - 1) // 2, 0)
        self.use_mdn = use_mdn
        self.split = split
        self.relation = relation

        # Spatial Edge
        self.gcns = nn.ModuleList()
        for r in range(self.relation):
            self.gcns.append(MultiRelationalGCN(in_channels, out_channels, kernel_size[1], relation=(len(split[r]))))

        # Temporal Edge
        self.tcn = nn.Sequential(
            nn.PReLU(),
            nn.Conv2d(out_channels, out_channels, (kernel_size[0], 1), (stride, 1), padding),
            nn.Dropout(dropout, inplace=True),
        )

        # Residual connection
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
            )

        self.prelu = nn.PReLU()

    def forward(self, x, A):
        res = self.residual(x)

        A_r = torch.split(A, 1, dim=1)
        for r in range(self.relation):
            A_ = get_disentangled_adjacency_matrix(A_r[r].squeeze(dim=1), self.split[r])
            x_a, _ = self.gcns[r](x, A_)

            if r == 0:
                x_r = x_a
            else:
                x_r = x_r + x_a

        x = self.tcn(x_r) + res

        if not self.use_mdn:
            x = self.prelu(x)

        return x, A
