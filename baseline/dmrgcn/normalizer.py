# Baseline model for "Disentangled Multi-Relational Graph Convolutional Network for Pedestrian Trajectory Prediction"
# Source-code referred from DMRGCN at https://github.com/InhwanBae/DMRGCN/blob/main/model/normalizer.py

# Adjacency matrix normalization implementation in PyTorch.
# The original code is based on Networkx library.

import torch


def normalized_adjacency_matrix(A):
    r"""Returns the normalized Adjacency matrix."""

    node_degrees = A.sum(-1).unsqueeze(dim=-1)
    degs_inv_sqrt = torch.pow(node_degrees, -0.5)
    degs_inv_sqrt[torch.isinf(degs_inv_sqrt)] = 0
    norm_degs_matrix = torch.eye(A.size(-1)).cuda() * degs_inv_sqrt
    return norm_degs_matrix @ A @ norm_degs_matrix


def normalized_adjacency_tilde_matrix(A):
    r"""Returns the normalized Adjacency tilde (A~) matrix."""

    A_t = A + torch.eye(A.size(-1)).cuda()
    return normalized_adjacency_matrix(A_t)


def normalized_laplacian_matrix(A):
    r"""Returns the normalized Laplacian matrix."""

    return torch.eye(A.size(-1)).cuda() - normalized_adjacency_matrix(A)


def normalized_laplacian_tilde_matrix(A):
    r"""Returns the normalized Laplacian tilde (L~) matrix."""

    A_t = A + torch.eye(A.size(-1)).cuda()
    return torch.eye(A_t.size(-1)).cuda() - normalized_adjacency_matrix(A_t)
