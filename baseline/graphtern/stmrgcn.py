import torch
import torch.nn as nn
from .dropedge import drop_edge
from .normalizer import normalized_adjacency_tilde_matrix


class MultiRelationalGCN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, t_kernel_size=1, t_stride=1, t_padding=0, t_dilation=1, bias=True, relation=2):
        super().__init__()
        self.kernel_size = kernel_size
        self.relation = relation
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels * relation, kernel_size=(t_kernel_size, 1), padding=(t_padding, 0), stride=(t_stride, 1), dilation=(t_dilation, 1), bias=bias)

    def forward(self, x, A):
        assert A.size(0) == x.size(0)
        assert A.size(1) == self.relation
        assert A.size(2) == self.kernel_size

        x = self.conv(x)
        x = x.view(x.size(0), self.relation, self.out_channels, x.size(-2), x.size(-1))
        x = torch.einsum('nrtwv,nrctv->nctw', normalized_adjacency_tilde_matrix(drop_edge(A, 0.8, self.training)), x)
        return x.contiguous(), A


class st_mrgcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, use_mdn=True, stride=1, dropout=0, residual=True, relation=2):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1

        padding = ((kernel_size[0] - 1) // 2, 0)
        self.use_mdn = use_mdn
        self.relation = relation
        self.prelu = nn.PReLU()
        self.gcn = MultiRelationalGCN(in_channels, out_channels, kernel_size[1], relation=self.relation)
        self.tcn = nn.Sequential(nn.PReLU(),
                                 nn.Conv2d(out_channels, out_channels, (kernel_size[0], 1), (stride, 1), padding),
                                 nn.Dropout(dropout, inplace=True),)

        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),)

    def forward(self, x, A):
        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x) + res

        if not self.use_mdn:
            x = self.prelu(x)

        return x, A


class epcnn(nn.Module):
    def __init__(self, obs_seq_len, pred_seq_len, in_channels, out_channels, n_tpcn=1, c_ksize=3, n_cpcn=1, t_ksize=3, dropout=0, residual=True):
        super().__init__()

        # NTCV
        self.tpcns = nn.ModuleList()
        for i in range(0, n_tpcn - 1):
            self.tpcns.append(nn.Sequential(nn.Conv2d(obs_seq_len, obs_seq_len, c_ksize, padding=c_ksize//2, padding_mode='replicate'),
                                            nn.PReLU(),
                                            nn.Dropout(dropout, inplace=True), ))
        self.tpcns.append(nn.Sequential(nn.Conv2d(obs_seq_len, pred_seq_len, c_ksize, padding=c_ksize//2, padding_mode='replicate'),
                                        nn.PReLU(),
                                        nn.Dropout(dropout, inplace=True), ))

        # NCTV
        self.cpcns = nn.ModuleList()
        for i in range(0, n_cpcn - 1):
            self.cpcns.append(nn.Sequential(nn.Conv2d(in_channels, in_channels, t_ksize, padding=t_ksize//2, padding_mode='replicate'),
                                            nn.PReLU(),
                                            nn.Dropout(dropout, inplace=True), ))
        self.cpcns.append(nn.Sequential(nn.Conv2d(in_channels, out_channels, t_ksize, padding=t_ksize//2, padding_mode='replicate'),
                                        nn.PReLU(),
                                        nn.Dropout(dropout, inplace=True), ))

        if not residual:
            self.residual = lambda x: 0
        elif obs_seq_len == pred_seq_len and in_channels == out_channels:
            self.residual = lambda x: x
        elif obs_seq_len == pred_seq_len:
            self.rescconv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1),)
            self.residual = lambda x: self.rescconv(x.permute(0, 2, 1, 3).contiguous()).permute(0, 2, 1, 3).contiguous()
        elif in_channels == out_channels:
            self.restconv = nn.Sequential(nn.Conv2d(obs_seq_len, pred_seq_len, kernel_size=1),)
            self.residual = lambda x: self.restconv(x)
        else:
            self.rescconv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1),)
            self.restconv = nn.Sequential(nn.Conv2d(obs_seq_len, pred_seq_len, kernel_size=1),)
            self.residual = lambda x: self.rescconv(self.restconv(x).permute(0, 2, 1, 3).contiguous()).permute(0, 2, 1, 3).contiguous()

    def forward(self, x):
        # residual
        res = self.residual(x)

        # time-wise
        for i in range(len(self.tpcns)):
            x = self.tpcns[i](x)

        # channel-wise
        x = x.permute(0, 2, 1, 3).contiguous()
        for i in range(len(self.cpcns)):
            x = self.cpcns[i](x)
        x = x.permute(0, 2, 1, 3).contiguous()

        return x + res


class trcnn(nn.Module):
    def __init__(self, total_seq_len, pred_seq_len, in_channels, out_channels, n_tpcn=1, c_ksize=3, n_cpcn=1, t_ksize=3, dropout=0, residual=True):
        super().__init__()

        # NTCV
        self.tpcns = nn.ModuleList()
        for i in range(0, n_tpcn-1):
            self.tpcns.append(nn.Sequential(nn.Conv2d(total_seq_len, total_seq_len, c_ksize, padding=c_ksize//2, padding_mode='replicate'),
                                            nn.PReLU(),
                                            nn.Dropout(dropout, inplace=True),))
        self.tpcns.append(nn.Sequential(nn.Conv2d(total_seq_len, pred_seq_len, c_ksize, padding=c_ksize//2, padding_mode='replicate'),
                                        nn.PReLU(),
                                        nn.Dropout(dropout, inplace=True),))

        # NCTV
        self.cpcns = nn.ModuleList()
        for i in range(0, n_cpcn-1):
            self.cpcns.append(nn.Sequential(nn.Conv2d(in_channels, in_channels, t_ksize, padding=t_ksize//2, padding_mode='replicate'),
                                            nn.PReLU(),
                                            nn.Dropout(dropout, inplace=True),))
        self.cpcns.append(nn.Sequential(nn.Conv2d(in_channels, out_channels, t_ksize, padding=t_ksize//2, padding_mode='replicate'),
                                        nn.PReLU(),
                                        nn.Dropout(dropout, inplace=True),))

        if not residual:
            self.residual = lambda x: 0
        elif total_seq_len == pred_seq_len:
            self.residual = lambda x: x
        else:
            k_size = total_seq_len - pred_seq_len + 1
            self.resconv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=(k_size, 1)),)
            self.residual = lambda x: self.resconv(x.permute(0, 2, 1, 3).contiguous()).permute(0, 2, 1, 3).contiguous()

    def forward(self, x):
        # residual
        res = self.residual(x)

        # time-wise
        for i in range(len(self.tpcns)):
            x = self.tpcns[i](x)

        # channel-wise
        x = x.permute(0, 2, 1, 3).contiguous()
        for i in range(len(self.cpcns)):
            x = self.cpcns[i](x)
        x = x.permute(0, 2, 1, 3).contiguous()

        return x + res
