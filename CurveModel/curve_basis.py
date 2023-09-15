import numpy as np
import torch


def torch_binom(n, k):
    mask = n.detach() >= k.detach()
    n = mask * n
    k = mask * k
    a = torch.lgamma(n + 1) - torch.lgamma((n - k) + 1) - torch.lgamma(k + 1)
    return torch.exp(a) * mask


def torch_pactorial(n):
    return torch.lgamma(n + 1).exp()


def irwin_hall_pdf(n, x):
    # https://en.wikipedia.org/wiki/Irwin%E2%80%93Hall_distribution
    k = torch.arange(0, n+1, 1, dtype=torch.float)
    n_ = torch.ones(n+1) * n
    comb = torch_binom(n_, k)
    sgn = (x - k)
    sgn_ = torch.zeros(n+1)
    eps = 1e-4
    sgn_[eps <= sgn] = 1
    sgn_[sgn <= -eps] = -1
    sigma = (torch.FloatTensor([-1]) ** k) * comb * ((x - k) ** (n-1)) * sgn_
    return sigma.sum() / (2 * torch_pactorial(torch.FloatTensor([n])-1))


def bezier_basis(degree=3, step=13):
    """Basis function for Bézier curve"""
    index = torch.linspace(0, 1, steps=step, dtype=torch.float).repeat(degree + 1, 1)
    i = torch.arange(0, degree + 1, 1, dtype=torch.float)
    binomial_coefficient = torch_binom(torch.ones(degree + 1) * degree, i)
    bernstein_basis_polynomial = binomial_coefficient * (index.T ** i) * ((1 - index.T) ** i.flip(0))
    return bernstein_basis_polynomial.detach()


def bspline_basis(cpoint=7, degree=2, step=13):
    """Piecewise polynomial function for basis-spline"""
    from scipy.interpolate import BSpline
    cpoint += 1
    steps = np.linspace(0., 1., step)
    knot = cpoint - degree + 1
    knots_qu = np.concatenate([np.zeros(degree), np.linspace(0, 1, knot), np.ones(degree)])
    bs = np.zeros([step, cpoint])
    for i in range(cpoint):
        bs[:, i] = BSpline(knots_qu, (np.arange(cpoint) == i).astype(float), degree, extrapolate=False)(steps)
    return torch.FloatTensor(bs)
