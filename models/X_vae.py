import torch
from models import BaseVAE
from torch import nn
from torch.nn import functional as F
from .types_ import *


class XrepVAE(BaseVAE):

    num_iter = 0 # Global static variable to keep track of iterations

    def __init__(self,
                 in_channels: int,
                 num_features: int,
                 content_added_dim: int,
                 style_added_dim: int,
                 hidden_dims: List = None,
                 beta: int = 4,
                 gamma:float = 1000.,
                 max_capacity: int = 25,
                 Capacity_max_iter: int = 1e5,
                 loss_type:str = 'B',
                 attr_weight: int = 0.5,
                 **kwargs) -> None:
        super(XrepVAE, self).__init__()

        self.num_features = num_features
        self.content_latent_dim = num_features - 1 + content_added_dim
        self.style_latent_dim = 1 + style_added_dim
        self.hidden_dims = hidden_dims
        self.beta = beta
        self.gamma = gamma
        self.loss_type = loss_type
        self.C_max = torch.Tensor([max_capacity])
        self.C_stop_iter = Capacity_max_iter
        self.attr_weight = attr_weight

        modules = []
        if self.hidden_dims is None:
            self.hidden_dims = [16, 32, 64, 128, 256, 512]

        # Build Content Encoder
        for h_dim in self.hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.content_encoder = nn.Sequential(*modules)
        self.content_fc_mu = nn.Linear(self.hidden_dims[-1]*4, self.content_latent_dim)
        self.content_fc_var = nn.Linear(self.hidden_dims[-1]*4, self.content_latent_dim)

        # Build Style Encoder
        self.style_encoder = nn.Sequential(*modules)
        self.style_fc_mu = nn.Linear(self.hidden_dims[-1]*4, self.style_latent_dim)
        self.style_fc_var = nn.Linear(self.hidden_dims[-1]*4, self.style_latent_dim)


        # Build Decoder
        modules = []
        self.decoder_input = nn.Linear(self.content_latent_dim + self.style_latent_dim, self.hidden_dims[-1] * 4)

        self.hidden_dims.reverse()

        for i in range(len(self.hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(self.hidden_dims[i],
                                       self.hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(self.hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )



        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(self.hidden_dims[-1],
                               self.hidden_dims[-1],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm2d(self.hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(self.hidden_dims[-1], out_channels=3,
                      kernel_size=3, padding=1),
            nn.Tanh())

    def encode(self, input: Tensor, encoder: str = 'content') -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        if encoder == 'content':
            result = self.content_encoder(input)
            result = torch.flatten(result, start_dim=1)
            mu = self.content_fc_mu(result)
            log_var = self.content_fc_var(result)
        elif encoder == 'style':
            result = self.style_encoder(input)
            result = torch.flatten(result, start_dim=1)
            mu = self.style_fc_mu(result)
            log_var = self.style_fc_var(result)
        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Will a single z be enough ti compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        content_mu, content_log_var = self.encode(input, "content")
        content_z = self.reparameterize(content_mu, content_log_var)

        style_mu, style_log_var = self.encode(input, "style")
        style_z = self.reparameterize(style_mu, style_log_var)
        z = torch.cat((content_z, style_z), 1)
        return [self.decode(z), input, content_mu, content_log_var, style_mu, style_log_var, z]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        self.num_iter += 1
        recons = args[0]
        input = args[1]
        content_mu = args[2]
        content_log_var = args[3]
        style_mu = args[4]
        style_log_var = args[5]
        z = args[6]

        # attribute loss
        attr_list = []
        for i in range(self.num_features):
            if i == 20:  # Male attribute:
                continue
            attr_list.append(kwargs['labels'][:, i].float())
        assert len(attr_list) == self.num_features - 1

        content_criterion = torch.nn.BCEWithLogitsLoss()
        style_criterion = torch.nn.BCEWithLogitsLoss()

        content_attr_loss = sum([content_criterion(z[:, i], attr_list[i]) for i in range(len(attr_list))])
        style_attr_loss = style_criterion(z[:, 39], kwargs['labels'][:, 20].float())
        attr_loss = content_attr_loss + style_attr_loss

        # reconstruction loss
        recons_loss = F.mse_loss(recons, input)

        # kld loss
        kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset
        content_kld_loss = torch.mean(-0.5 * torch.sum(1 + content_log_var - content_mu ** 2 - content_log_var.exp(),
                                                       dim=1), dim=0)
        style_kld_loss = torch.mean(-0.5 * torch.sum(1 + style_log_var - style_mu ** 2 - style_log_var.exp(), dim=1),
                                    dim=0)
        kld_loss = content_kld_loss + style_kld_loss

        if self.loss_type == 'H': # https://openreview.net/forum?id=Sy2fzU9gl
            loss = (attr_loss / (self.attr_weight * (self.content_latent_dim + self.style_latent_dim)))\
                   + recons_loss + (self.beta * kld_weight * kld_loss)
        elif self.loss_type == 'B': # https://arxiv.org/pdf/1804.03599.pdf
            self.C_max = self.C_max.to(input.device)
            C = torch.clamp(self.C_max/self.C_stop_iter * self.num_iter, 0, self.C_max.data[0])
            loss = attr_loss + recons_loss + self.gamma * kld_weight * (kld_loss - C).abs()
        else:
            raise ValueError('Undefined loss type.')

        return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'KLD':kld_loss, 'AttrLoss': attr_loss}

    def sample(self,
               num_samples:int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """

        # uniform between [-3, 3]
        z = (3 - (-3)) * torch.rand(num_samples, self.content_latent_dim + self.style_latent_dim) + (-3)
        z = z.to(current_device)
        if "latent_var" in kwargs:
            samples = []
            for latent_val in range(-3, 4):
                z[:, kwargs['latent_var']] = torch.ones_like(z[:, kwargs['latent_var']]) * latent_val
                samples.append(((self.decode(z)), latent_val))

        else:
            samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]