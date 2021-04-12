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


class SequentialLinearModule(nn.Module):


    def __init__(self, in_dim, out_dim, layer_sizes, activation, out_activation):

        super().__init__()
        
        # layers is the list that contains the sizes of linear layers between input and output
        layer_sizes.append(out_dim)
        layers = [] # list of the modules that we put into sequential
        layers.append(nn.Linear(in_dim, layer_sizes[0]))
        for i in range(len(layer_sizes)-1):
            layers.append(activation()) # activation after the last linear
            layers.append(nn.BatchNorm1d(layer_sizes[i])) # batchnorm
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1])) # linear
        layers.append(out_activation()) # final activation
        self.f = nn.Sequential(*layers) # combine everything

    def forward(self, x):
        return self.f(x)
    
class Generator(SequentialLinearModule):
    def __init__(self, noise_dim, n_dim, layer_sizes):
        super().__init__(noise_dim, n_dim, layer_sizes, nn.ReLU, nn.Identity)
        self.noise_dim = noise_dim
        
class Discriminator(SequentialLinearModule):
    def __init__(self, n_dim, layers):
        super().__init__(n_dim, 1, layer_sizes, nn.LeakyReLU, nn.Sigmoid)
        
class Critic(SequentialLinearModule):
    def __init__(self, n_dim, layer_sizes):
        super().__init__(n_dim, 1, layer_sizes, nn.LeakyReLU, nn.LeakyReLU)
        
class ChainedGenerator(nn.Module):
    # note that layer_sizes are for a single variable, thus the generator itself is going to be n_dim times larger
    def __init__(self, noise_dim, n_dim, layer_sizes):
        
        super().__init__()
        
        self.n_dim = n_dim
        self.noise_dim = noise_dim
        
        activation = nn.ReLU
        out_activation = nn.Identity
        
        self.layers = nn.ModuleList()
        for i in range(n_dim):
            self.layers.append(SequentialLinearModule(noise_dim//n_dim + i,1,layer_sizes, activation, out_activation))
                
    def forward(self, noise):
        noise = noise.view(noise.shape[0], self.n_dim, self.noise_dim//self.n_dim)
        noise_list = [noise[:,i,:] for i in range(self.n_dim)]
        variables = []
        for i in range(self.n_dim):
            layer_input = torch.cat([noise_list[i], *variables], dim=1)
            layer_output = self.layers[i](layer_input)
            variables.append(layer_output)
        return torch.cat(variables, dim=1)
    
    def __repr__(self):
        
        return '\n'.join([str(i) for i in self.layers])