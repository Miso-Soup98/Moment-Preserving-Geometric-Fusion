import torch
import numpy as np
from scipy.spatial.transform import Rotation

def kabsch_rotation(P, Q):
    C = np.dot(np.transpose(P), Q)
    V, S, W = np.linalg.svd(C)
    d = np.linalg.det(V) * np.linalg.det(W) < 0.0
    if d:
        S[-1] = -S[-1]
        V[:, -1] = -V[:, -1]
    U = np.dot(V, W)
    return U

def kabsch(a, b):
    a, b = (np.array(a), np.array(b))
    a_mean = np.mean(a, axis=0)
    b_mean = np.mean(b, axis=0)
    a_c = a - a_mean
    b_c = b - b_mean
    rotation = kabsch_rotation(a_c, b_c)
    t = b_mean - np.dot(a_mean, rotation)
    a_aligned = np.dot(a, rotation) + t
    return (a_aligned, rotation, t)

def compute_rmsd(a, b, aligned=False):
    if aligned:
        a_aligned = a
    else:
        a_aligned, _, _ = kabsch(a, b)
    dist = np.sum((a_aligned - b) ** 2, axis=-1)
    rmsd = np.sqrt(dist.sum() / a.shape[0])
    return float(rmsd)

def kabsch_torch(A, B, requires_grad=False):
    a_mean = A.mean(axis=0)
    b_mean = B.mean(axis=0)
    A_c = A - a_mean
    B_c = B - b_mean
    H = A_c.T.mm(B_c)
    if requires_grad:
        assert not torch.isnan(H).any()
        U, S, Vt = torch.linalg.svd(H)
        num_it = 0
        while torch.min(S) < 0.001 or torch.min(torch.abs((S ** 2).view(1, 3) - (S ** 2).view(3, 1) + torch.eye(3).to(S.device))) < 0.01:
            H = H + torch.rand(3, 3).to(H.device) * torch.eye(3).to(H.device)
            U, S, Vt = torch.linalg.svd(H)
            num_it += 1
            if num_it > 10:
                raise RuntimeError('SVD consistently numerically unstable! Exitting ... ')
    else:
        U, S, Vt = torch.linalg.svd(H)
    V = Vt.T
    d = torch.linalg.det(U) * torch.linalg.det(V) < 0.0
    if d:
        SS = torch.diag(torch.tensor([1.0 for _ in range(len(U) - 1)] + [-1.0], device=U.device, dtype=U.dtype))
        U = U @ SS
    R = V.mm(U.T)
    t = b_mean[None, :] - R.mm(a_mean[None, :].T).T
    t = t.T.squeeze()
    return (R.mm(A.T).T + t, R, t)

def batch_kabsch_torch(A, B):
    a_mean = A.mean(dim=1, keepdims=True)
    b_mean = B.mean(dim=1, keepdims=True)
    A_c = A - a_mean
    B_c = B - b_mean
    H = torch.bmm(A_c.transpose(1, 2), B_c)
    U, S, Vt = torch.linalg.svd(H)
    V = Vt.transpose(1, 2)
    d = (torch.linalg.det(U) * torch.linalg.det(V) < 0.0).long()
    nSS = torch.diag(torch.tensor([1.0 for _ in range(len(U))], device=U.device, dtype=U.dtype))
    SS = torch.diag(torch.tensor([1.0 for _ in range(len(U) - 1)] + [-1.0], device=U.device, dtype=U.dtype))
    bSS = torch.stack([nSS, SS], dim=0)[d]
    U = torch.bmm(U, bSS)
    R = torch.bmm(V, U.transpose(1, 2))
    t = b_mean - torch.bmm(R, a_mean.transpose(1, 2)).transpose(1, 2)
    A_aligned = torch.bmm(R, A.transpose(1, 2)).transpose(1, 2) + t
    return (A_aligned, R, t)
