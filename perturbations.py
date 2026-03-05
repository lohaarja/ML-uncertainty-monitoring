import torch
import torchvision.transforms.functional as F
def gaussian_noise(x, sigma):
    return x + sigma * torch.randn_like(x)
def blur(x, kernel_size):
    return F.gaussian_blur(x, kernel_size=[kernel_size, kernel_size])
def brightness_shift(x, factor):
    return torch.clamp(x * factor, 0, 1)
def apply_perturbation(x, step):
    x = gaussian_noise(x, sigma = 0.03 * step)
    if step % 3 == 0:
        x = blur(x, kernel_size=3)
    if step % 5 == 0:
        x = brightness_shift(x, 1 - 0.05 * step)
    return torch.clamp(x, 0, 1)