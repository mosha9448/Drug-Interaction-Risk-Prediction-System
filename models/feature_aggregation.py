import torch


def aggregate_features(low_feat, high_feat):
    """
    Kernel-wise feature aggregation
    """

    fused = torch.cat([low_feat, high_feat], dim=1)

    return fused