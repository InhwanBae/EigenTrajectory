import sys
import torch


def curve_fitting(traj, basis):
    n_ped, t_traj, dim = traj.shape
    n_cp = basis.size(1)

    class Model(torch.nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            cp_init = torch.FloatTensor(n_ped, n_cp, dim)
            cp_init[:, 0], cp_init[:, -1] = traj[:, 0], traj[:, -1]
            for i in range(1, n_cp):
                cp_init[:, i] = cp_init[:, i - 1] + (traj[:, -1] - traj[:, 0]) / (n_cp - 1)
            self.cp = torch.nn.Parameter(cp_init)  # Fast convergence
            # self.cp = torch.nn.Parameter(torch.FloatTensor(n_ped, n_cp, dim))
            # torch.nn.init.xavier_uniform_(self.cp, gain=1.0)

        def forward(self):
            recon = (self.cp.transpose(1, 2) @ basis.T).transpose(1, 2)
            loss = (recon - traj).norm(p=2, dim=-1).mean()
            return recon, loss

    model = Model()
    model.train()
    if torch.cuda.is_available():
        model = model.cuda()
        traj, basis = traj.cuda(), basis.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    recon_best, loss_best = None, 1e8
    for _ in range(100000):
        recon, loss = model()
        loss.backward()
        optimizer.step()
        sys.stdout.write('\r\033Curve Fitting... loss={:.4f}'.format(loss.item()))
        if loss.item() < loss_best:
            recon_best = recon.detach().cpu()
            loss_best = loss.item()

    sys.stdout.write('\r\033Curve Fitting... Done.\n')
    return recon_best
