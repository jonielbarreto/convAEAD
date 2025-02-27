import torch
import torch.nn as nn
import torch.nn.functional as F
from convlstm import ConvLSTM

class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dim=512):
        super(VariationalAutoencoder, self).__init__()
        self.latent_dim = latent_dim

        # ConvLSTM Encoder
        self.conv_lstm = ConvLSTM(
            input_dim=3,  # Number of input channels (RGB images)
            hidden_dim=64,  # Reduced hidden dimension to avoid excessive memory usage
            kernel_size=(3, 3),
            num_layers=1,
            batch_first=True,
            bias=True,
            return_all_layers=False
        )

        # Encoder (Conv2d layers)
        self.encoder = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # Input channels: 64 (from ConvLSTM)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Conv2d(256, 256, kernel_size=4, stride=4, padding=1),
        )

        # Fully connected layers for latent space
        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(256 * 8 * 8, latent_dim)
        self.fc_logvar = nn.Linear(256 * 8 * 8, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 256 * 8 * 8),
            nn.Unflatten(1, (256, 8, 8)),  # Shape: [batch_size, 256, 8, 8]
            
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1, output_padding=0),  #Output: [B, 256, 16, 16]
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.5),
            
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, output_padding=0),  #Output: [B, 128, 32, 32]
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.5),
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, output_padding=0),  #Output: [B, 64, 64, 64]
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.5),
            
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, output_padding=0),  #Output: [B, 32, 128, 128]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=1, padding=1),  #Output: [B, 3, 128, 128]
            nn.BatchNorm2d(3),
            nn.Tanh()
        )

    def encode(self, x):
        # Pass input through ConvLSTM
        _, last_states = self.conv_lstm(x)
        x = last_states[0][0]  # Extract the last hidden state from ConvLSTM

        # Pass through encoder
        for layer in self.encoder:
            x = layer(x)
        x = F.normalize(x, p=2, dim=1)
        x = self.flatten(x)

        # Compute mu and logvar
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        for i, layer in enumerate(self.decoder):
            z = layer(z)
        output = (z + 1) / 2
        #print(f"Output shape in decode: {output.shape}")  # Should be [batch_size, 3, 128, 128]
        return output

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        z = (z - z.mean()) / (z.std() + 1e-6)
        return self.decode(z), mu, logvar

