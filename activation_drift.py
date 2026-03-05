import torch

def activation_drift(prev_act, curr_act):
    return torch.norm(curr_act - prev_act, p=2).item()
