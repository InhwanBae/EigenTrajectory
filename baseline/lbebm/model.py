# Baseline model for "Trajectory Prediction with Latent Space Energy-Based Model"
# Source-code referred from LB-EBM at https://github.com/bpucla/lbebm/blob/main/lbebm_univ.py

import random
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=(1024, 512), activation='relu', discrim=False, dropout=-1):
        super(MLP, self).__init__()
        dims = []
        dims.append(input_dim)
        dims.extend(hidden_size)
        dims.append(output_dim)
        self.layers = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i + 1]))

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()

        self.sigmoid = nn.Sigmoid() if discrim else None
        self.dropout = dropout

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            if i != len(self.layers) - 1:
                x = self.activation(x)
                if self.dropout != -1:
                    x = nn.Dropout(min(0.1, self.dropout / 3) if i == 1 else self.dropout)(x)
            elif self.sigmoid:
                x = self.sigmoid(x)
        return x


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, input_memory):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = input_memory
        self.position = (self.position + 1) % self.capacity

    def sample(self, n=100):
        samples = random.sample(self.memory, n)
        return torch.cat(samples)

    def __len__(self):
        return len(self.memory)


class LBEBM(nn.Module):
    def __init__(self, enc_past_size, enc_dest_size, enc_latent_size, dec_size, predictor_size, fdim, zdim, sigma,
                 past_length, future_length, args):
        super(LBEBM, self).__init__()
        self.zdim = zdim
        self.sigma = sigma
        self.nonlocal_pools = args.nonlocal_pools
        non_local_dim = args.non_local_dim
        non_local_phi_size = args.non_local_phi_size
        non_local_g_size = args.non_local_g_size
        non_local_theta_size = args.non_local_theta_size

        self.encoder_past = MLP(input_dim=past_length * 2, output_dim=fdim, hidden_size=enc_past_size)
        self.encoder_dest = MLP(input_dim=len(args.sub_goal_indexes) * 2, output_dim=fdim, hidden_size=enc_dest_size)
        self.encoder_latent = MLP(input_dim=2 * fdim, output_dim=2 * zdim, hidden_size=enc_latent_size)
        self.decoder = MLP(input_dim=fdim + zdim, output_dim=len(args.sub_goal_indexes) * 2, hidden_size=dec_size)
        self.predictor = MLP(input_dim=2 * fdim, output_dim=2 * (future_length), hidden_size=predictor_size)

        self.non_local_theta = MLP(input_dim=fdim, output_dim=non_local_dim, hidden_size=non_local_theta_size)
        self.non_local_phi = MLP(input_dim=fdim, output_dim=non_local_dim, hidden_size=non_local_phi_size)
        self.non_local_g = MLP(input_dim=fdim, output_dim=fdim, hidden_size=non_local_g_size)

        self.EBM = nn.Sequential(
            nn.Linear(zdim + fdim, 200),
            nn.GELU(),
            nn.Linear(200, 200),
            nn.GELU(),
            nn.Linear(200, args.ny),
        )

        self.replay_memory = ReplayMemory(args.memory_size)
        self.args = args

    def forward(self, x, dest=None, mask=None, iteration=1, y=None):

        ftraj = self.encoder_past(x)

        if mask:
            for _ in range(self.nonlocal_pools):
                ftraj = self.non_local_social_pooling(ftraj, mask)

        if self.training:
            pcd = True if len(self.replay_memory) == self.args.memory_size else False
            if pcd:
                z_e_0 = self.replay_memory.sample(n=ftraj.size(0)).clone().detach().cuda()
            else:
                z_e_0 = self.sample_p_0(n=ftraj.size(0), nz=self.zdim)
            z_e_k, _ = self.sample_langevin_prior_z(Variable(z_e_0), ftraj, pcd=pcd, verbose=(iteration % 1000 == 0))
            for _z_e_k in z_e_k.clone().detach().cpu().split(1):
                self.replay_memory.push(_z_e_k)
        else:
            z_e_0 = self.sample_p_0(n=ftraj.size(0), nz=self.zdim)
            z_e_k, _ = self.sample_langevin_prior_z(Variable(z_e_0), ftraj, pcd=False, verbose=(iteration % 1000 == 0),
                                                    y=y)
        z_e_k = z_e_k.cuda()

        if self.training:
            dest_features = self.encoder_dest(dest)
            features = torch.cat((ftraj, dest_features), dim=1)
            latent = self.encoder_latent(features)
            mu = latent[:, 0:self.zdim]
            logvar = latent[:, self.zdim:]

            var = logvar.mul(0.5).exp_()
            eps = torch.DoubleTensor(var.size()).normal_().cuda()
            z_g_k = eps.mul(var).add_(mu)
            z_g_k = z_g_k.cuda()

        if self.training:
            decoder_input = torch.cat((ftraj, z_g_k), dim=1)
        else:
            decoder_input = torch.cat((ftraj, z_e_k), dim=1)
        generated_dest = self.decoder(decoder_input)

        if self.training:
            generated_dest_features = self.encoder_dest(generated_dest)
            prediction_features = torch.cat((ftraj, generated_dest_features), dim=1)
            pred_future = self.predictor(prediction_features)

            en_pos = self.ebm(z_g_k, ftraj).mean()
            en_neg = self.ebm(z_e_k.detach().clone(), ftraj).mean()
            cd = en_pos - en_neg

            return generated_dest, mu, logvar, pred_future, cd, en_pos, en_neg, pcd

        return generated_dest

    def ebm(self, z, condition, cls_output=False):
        condition_encoding = condition.detach().clone()
        z_c = torch.cat((z, condition_encoding), dim=1)
        conditional_neg_energy = self.EBM(z_c)
        assert conditional_neg_energy.shape == (z.size(0), self.args.ny)
        if cls_output:
            return - conditional_neg_energy
        else:
            return - conditional_neg_energy.logsumexp(dim=1)

    def sample_p_0(self, n, nz=16):
        return self.args.e_init_sig * torch.randn(*[n, nz]).cuda()

    def sample_langevin_prior_z(self, z, condition, pcd=False, verbose=False, y=None):
        z = z.clone().detach()
        z.requires_grad = True
        _e_l_steps = self.args.e_l_steps_pcd if pcd else self.args.e_l_steps
        _e_l_step_size = self.args.e_l_step_size
        for i in range(_e_l_steps):
            if y is None:
                en = self.ebm(z, condition)
            else:
                en = self.ebm(z, condition, cls_output=True)[range(z.size(0)), y]
            z_grad = torch.autograd.grad(en.sum(), z)[0]

            z.data = z.data - 0.5 * _e_l_step_size * _e_l_step_size * \
                     (z_grad + 1.0 / (self.args.e_prior_sig * self.args.e_prior_sig) * z.data)
            if self.args.e_l_with_noise:
                z.data += _e_l_step_size * torch.randn_like(z).data

            if (i % 5 == 0 or i == _e_l_steps - 1) and verbose:
                if y is None:
                    print('Langevin prior {:3d}/{:3d}: energy={:8.3f}'.format(i + 1, _e_l_steps, en.sum().item()))
                else:
                    print('Conditional Langevin prior {:3d}/{:3d}: energy={:8.3f}'.format(i + 1, _e_l_steps,
                                                                                          en.sum().item()))

            z_grad_norm = z_grad.view(z_grad.size(0), -1).norm(dim=1).mean()

        return z.detach(), z_grad_norm

    def predict(self, past, generated_dest):
        ftraj = self.encoder_past(past)
        generated_dest_features = self.encoder_dest(generated_dest)
        prediction_features = torch.cat((ftraj, generated_dest_features), dim=1)
        interpolated_future = self.predictor(prediction_features)

        return interpolated_future

    def non_local_social_pooling(self, feat, mask):
        theta_x = self.non_local_theta(feat)
        phi_x = self.non_local_phi(feat).transpose(1, 0)
        f = torch.matmul(theta_x, phi_x)
        f_weights = F.softmax(f, dim=-1)
        f_weights = f_weights * mask
        f_weights = F.normalize(f_weights, p=1, dim=1)
        pooled_f = torch.matmul(f_weights, self.non_local_g(feat))

        return pooled_f + feat
