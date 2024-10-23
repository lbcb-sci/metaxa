import torch
import torch.nn.functional as F


def nt_xent_loss(preds, targets, temperature=0.1):
    N = preds.shape[0]

    preds = F.normalize(preds, dim=1)
    targets = F.normalize(targets, dim=1)

    representations = torch.cat([preds, targets], dim=0)

    # Compute similarity matrix
    similarity_matrix = torch.matmul(
        representations, representations.T
    )  # Shape: (2N, 2N)

    # Remove self-similarities
    mask = torch.eye(2 * N, device=representations.device).bool()
    similarity_matrix = similarity_matrix.masked_fill(mask, float('-inf'))

    # Positive pairs are [i, i+N] and [i+N, i]
    positives = torch.cat(
        [torch.diag(similarity_matrix, N), torch.diag(similarity_matrix, -N)], dim=0
    )

    # Compute numerator and denominator
    numerator = torch.exp(positives / temperature)
    denominator = torch.sum(torch.exp(similarity_matrix / temperature), dim=1)

    # Compute loss
    loss = -torch.log(numerator / denominator)
    loss = loss.mean()
    return loss


def nt_xent_loss2(preds, targets, temperature=0.2):
    N = preds.shape[0]

    reps = torch.cat([preds, targets], dim=0)
    similarity_matrix = F.cosine_similarity(reps[None, :, :], reps[:, None, :], dim=-1)

    # Remove self-similarities
    mask = torch.eye(2 * N, device=reps.device).bool()
    similarity_matrix.masked_fill_(mask, float('-inf'))

    tt = torch.roll(torch.arange(2 * N, device=reps.device), N)
    loss = F.cross_entropy(similarity_matrix / temperature, tt)

    return loss


def barlow_twins_loss(
    z_a: torch.Tensor, z_b: torch.Tensor, lbda: float = 5e-3
) -> torch.Tensor:
    N, D = z_a.size()

    z_a_norm = (z_a - z_a.mean(0)) / z_a.std(0)
    z_b_norm = (z_b - z_b.mean(0)) / z_b.std(0)

    c = torch.matmul(z_a_norm.T, z_b_norm) / N
    c_diff = (c - torch.eye(D, device=c.device)).pow(2)

    off_diag_mask = ~torch.eye(D, dtype=bool)
    c_diff[off_diag_mask].mul_(lbda)
    loss = c_diff.sum()

    return loss
