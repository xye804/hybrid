import torch
import torch.nn as nn
import torch.nn.functional as F


def bone_length_loss(pred, target):
    """
    pred, target: (B, T, J, 2 or 3)
    """

    POSE_PAIRS = [
        (1, 2),
        (2, 3),
        (3, 4),
        (1, 5),
        (5, 6),
        (6, 7),
        (10, 11),
        (11, 12),
        (5, 12),
        (9, 12),
        (2, 9),
        (13, 14),
    ]

    HAND_PAIRS = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (0, 5),
        (5, 6),
        (6, 7),
        (7, 8),
        (0, 9),
        (9, 10),
        (10, 11),
        (11, 12),
        (0, 13),
        (13, 14),
        (14, 15),
        (15, 16),
        (0, 17),
        (17, 18),
        (18, 19),
        (19, 20),
    ]

    bone_pairs = POSE_PAIRS + HAND_PAIRS

    def get_lengths(joints):
        return torch.stack(
            [
                torch.norm(joints[..., i, :] - joints[..., j, :], dim=-1)
                for (i, j) in bone_pairs
            ],
            dim=-1,
        )

    pred_lens = get_lengths(pred)
    target_lens = get_lengths(target)

    return F.mse_loss(pred_lens, target_lens)


def adaptive_l2_loss(error, gamma=0.5, c=1e-3):
    delta_sq = torch.mean(error**2, dim=(1, 2, 3), keepdim=False)
    p = 1.0 - gamma
    w = 1.0 / (delta_sq + c).pow(p)
    loss = delta_sq
    return (w.detach() * loss).mean()
