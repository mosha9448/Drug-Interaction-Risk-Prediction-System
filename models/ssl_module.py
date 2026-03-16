import torch
import torch.nn.functional as F


def ssl_loss(embeddings):
    """
    Simple self-supervised loss
    """

    similarity = torch.matmul(embeddings, embeddings.T)

    target = torch.eye(similarity.shape[0])

    loss = F.mse_loss(similarity, target)

    return loss