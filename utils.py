import torch
import torch.nn as nn

def gradient_penalty(critic, real, fake, device):
    batch_size, n_dim = real.shape
    epsilon = torch.rand((batch_size, 1)).repeat(1, n_dim).to(device)
    interpolated_images = real*epsilon + fake*(1-epsilon)

    mixed_scores = critic(interpolated_images)
    gradient = torch.autograd.grad(
        inpuits=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]

    print(gradient)