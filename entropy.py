import torch
import torch.nn.functional as F
def entropy(logits):
    probs = F.softmax(logits, dim=-1)
    return -(probs * torch.log(probs + 1e-8)).sum(dim=-1)