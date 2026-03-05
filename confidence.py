import torch.nn.functional as F

def confidence(logits):
    return F.softmax(logits, dim=-1).max(dim=-1).values