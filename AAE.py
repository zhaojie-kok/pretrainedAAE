import torch
import torch.nn as nn
import copy
# TODO: docstrings and train loop (can't use from toolbox)


class encoder(nn.Module):
    def __init__(self,
                 input_dim,
                 latent_dim,
                 n_classes,
                 hidden_dim=1,
                 n_layers=0,
                 dropout=0.2,
                 input_block=None,
                 hidden_block=None,
                 y_block=None,
                 z_block=None):
        super(encoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        # create structure
        if (input_block is None):
            conv_block = nn.Sequential(nn.Conv2d(3, 64, 3, stride=2),
                                       nn.MaxPool2d(3), nn.BatchNorm2d(64),
                                       nn.ReLU(),
                                       nn.Conv2d(64, 128, 3, stride=2),
                                       nn.ReLU(), nn.Dropout2d(dropout),
                                       nn.Flatten())
            self.input_block = nn.Sequential(
                conv_block,
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
            )
        else:
            self.input_block = input_block

        blocks = nn.ModuleList()
        if (hidden_block is None):
            hidden_block = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
            )

        for i in range(n_layers):
            blocks.append(copy.deepcopy(hidden_block))

        self.blocks = nn.Sequential(*blocks)

        if (y_block is None):
            self.y_block = nn.Sequential(nn.Dropout(dropout),
                                         nn.Linear(hidden_dim, n_classes),
                                         nn.Softmax())
        else:
            self.y_block = y_block

        if (z_block is None):
            self.z_block = nn.Sequential(nn.Dropout(dropout),
                                         nn.BatchNorm1d(hidden_dim),
                                         nn.Linear(hidden_dim, latent_dim))
        else:
            self.z_block = z_block

    def forward(self, input):
        x = self.input_block(input)
        x = self.blocks(x)

        z = self.z_block(x)
        y_ = self.y_block(x)

        return z, y_


class decoder(nn.Module):
    def __init__(self,
                 latent_dim,
                 output_dim,
                 hidden_dim=1,
                 n_layers=0,
                 dropout=0.5,
                 input_block=None,
                 hidden_block=None,
                 output_block=None):
        super(decoder, self).__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # create structure
        if (input_block is None):
            self.input_block = nn.Sequential(nn.Linear(latent_dim, hidden_dim),
                                             nn.ReLU(),
                                             nn.BatchNorm1d(hidden_dim))
        else:
            self.input_block = input_block

        if (hidden_block is None):
            hidden_block = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                         nn.ReLU(), nn.BatchNorm1d(hidden_dim))

        blocks = nn.ModuleList()
        for i in range(n_layers):
            blocks.append(copy.deepcopy(hidden_block))
        blocks.append(nn.Linear(hidden_dim, output_dim))
        self.blocks = nn.Sequential(*blocks)

        if (output_block is None):
            deconv_block = nn.Sequential(
                nn.ConvTranspose2d(128, 64, 3, stride=2), nn.ReLU(),
                nn.BatchNorm2d(64),
                nn.ConvTranspose2d(64, 32, 3, stride=2, padding=(1, 1)),
                nn.ReLU(),
                nn.ConvTranspose2d(32, 16, 3, stride=2, padding=(1, 1)),
                nn.ReLU(), nn.ConvTranspose2d(16, 3, 4, stride=2, padding=2))
            self.output_block = nn.Sequential(deconv_block, nn.Sigmoid())
        else:
            self.output_block = output_block

    def forward(self, input):
        x = self.input_block(input)
        x = self.blocks(x)
        # print(x.shape)
        x = x.view(x.size(0), 128, 2, 2)
        return self.output_block(x)


class discrim(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, dropout=0.2):
        super(discrim, self).__init__()
        self.input_block = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                         nn.Dropout(dropout), nn.ReLU())

        blocks = nn.ModuleList()
        for i in range(n_layers):
            blocks.append(
                nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU()))
        self.blocks = nn.Sequential(*blocks)

        self.out_block = nn.Sequential(nn.Dropout(dropout),
                                       nn.Linear(hidden_dim, 1), nn.Sigmoid())

    def forward(self, input):
        x = self.input_block(input)
        x = self.blocks(x)
        return self.out_block(x)


class rand_sampler():
    def __init__(self,
                 dim,
                 mu=None,
                 sig=None,
                 mode='cat',
                 prob_tensor=None,
                 portion=None):
        """
        dim: dimensionality for continuous random variable/num classes for cat
        mode
        """
        assert mode in ('cat', 'cont', 'multi')
        self.mode = mode
        self.dim = dim

        if mode == 'cont':
            self.mu = mu
            self.sig = sig
        elif mode == 'cat':
            prob_tensor = torch.ones(
                dim) / dim if prob_tensor is None else prob_tensor
            self.sampler = torch.distributions.OneHotCategorical(prob_tensor)
        elif mode == 'multi':
            self.portion = portion

    def sample(self, n_samples):
        if self.mode == 'cont':
            return self.mu + (torch.randn(n_samples, self.dim) * self.sig)
        elif self.mode == 'cat':
            return self.sampler.sample((n_samples, ))
        elif self.mode == 'multi':
            return torch.rand(n_samples, self.dim)


class semi_sup_AAE(nn.Module):
    def __init__(self, encoder, decoder, discrim_z, discrim_y, num_classes,
                 z_rand, y_rand):
        super(semi_sup_AAE, self).__init__()
        self.enc = encoder
        self.dec = decoder
        self.d_z = discrim_z
        self.d_y = discrim_y
        self.num_classes = num_classes
        self.z_rand = z_rand
        self.y_rand = y_rand

    def forward(self, input):
        z, y_ = self.enc(input)
        x_ = self.dec(torch.cat((z, y_), dim=1))
        return x_, z, y_

    def get_reconst(self, input):
        z, y_ = self.enc(input)
        x_ = self.dec(torch.cat((z, y_), dim=1))
        return x_

    def get_latents(self, input):
        z, y_ = self.enc(input)
        return z, y_

    def discriminate(self, input, mode):
        assert mode in 'yz'
        if mode == 'y':
            return self.d_y(input)
        elif mode == 'z':
            return self.d_z(input)

    def get_samples(self, n_samples):
        return self.z_rand.sample(n_samples), self.y_rand.sample(n_samples)


def make_model(input_shape,
               hidden_enc,
               enc_layers,
               latent_y,
               latent_z,
               hidden_dec,
               dec_layers,
               disc_y,
               disc_y_layers,
               disc_z,
               disc_z_layers,
               dropout_rate=0.1):
    enc = encoder(input_shape,
                  latent_z,
                  latent_y,
                  hidden_enc,
                  enc_layers,
                  dropout=dropout_rate)
    dec = decoder(latent_y + latent_z,
                  input_shape,
                  hidden_dec,
                  dec_layers,
                  dropout=dropout_rate)
    disc_z = discrim(latent_z, disc_z, disc_z_layers, dropout=dropout_rate)
    disc_y = discrim(latent_y, disc_y, disc_y_layers, dropout=dropout_rate)

    z_sampler = rand_sampler(latent_z, mu=0, sig=1, mode='cont')
    y_sampler = rand_sampler(
        latent_y, mode='multi',
        portion=3 / 14)  # categorical samples of y are terrible for ood

    return semi_sup_AAE(enc, dec, disc_z, disc_y, latent_y, z_sampler,
                        y_sampler)
