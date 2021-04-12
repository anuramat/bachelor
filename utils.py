import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from io import BytesIO

def gradient_penalty(critic, real, fake, device):
    # TODO learn what the fuck is going on
    batch_size, n_dim = real.shape
    epsilon = torch.rand((batch_size, 1)).repeat(1, n_dim).to(device)
    interpolated_images = real*epsilon + fake*(1-epsilon)

    mixed_scores = critic(interpolated_images)
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0] # shape is (batch_size, n_dim=5)

    gradient_norm = gradient.norm(2, dim=1) # (batch_size)

    return torch.mean((gradient_norm - 1) ** 2)

def draw_hist(real_data, gen, device, df):

    noise = torch.randn(real_data.shape[0], gen.noise_dim).to(device)
    fake = gen(noise).detach().cpu().numpy()
    real = real_data
    bins = 100

    plotnum = 1
    fig, ax = plt.subplots(figsize=(14, 70)) # size in inches
    for i in range(5):
        for j in range(i+1, 5):
            # (i,j) are the columns that we are going to draw a histogram of

            # real data
            plt.subplot(10, 2, plotnum)
            plt.title(f'real, {df.columns[i]}/{df.columns[j]}')
            plotnum += 1
            _, xedges, yedges, _ = plt.hist2d(real[:,i], real[:,j], bins=bins,)

            x_min = xedges[0]
            x_max = xedges[-1]
            y_min = yedges[0]
            y_max = yedges[-1]

            # fake data
            plt.subplot(10, 2, plotnum)
            plt.title(f'fake, {df.columns[i]}/{df.columns[j]}')
            plotnum += 1
            plt.hist2d(fake[:,i], fake[:,j], bins=bins, range = [[x_min,x_max], [y_min,y_max]])

    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close('all')
    buf.seek(0)
    return buf
