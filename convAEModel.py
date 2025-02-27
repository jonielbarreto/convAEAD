import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the autoencoder architecture
class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dim=512):
        super(VariationalAutoencoder, self).__init__()
        self.latent_dim = latent_dim

        #Encoder
        self.encoder = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(4, 3, 3), stride=2, padding=(0, 1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Conv2d(256, 512, kernel_size=4, stride=4, padding=1),
        )
        
        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(512*8*8, self.latent_dim)
        self.fc_logvar = nn.Linear(512*8*8, self.latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 512 * 8 * 8),  # Reshape to (batch_size, 512, 8, 8)
            nn.Unflatten(1, (512, 8, 8)),             # Unflatten to (batch_size, 512, 8, 8)
            
            # Upsample to (batch_size, 256, 16, 16)
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.5),
            
            # Upsample to (batch_size, 128, 32, 32)
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.5),
            
            # Upsample to (batch_size, 64, 64, 64)
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.5),
            
            # Upsample to (batch_size, 32, 128, 128)
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            #nn.LeakyReLU(0.5),
            
            # Upsample to (batch_size, 3, 256, 256)
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(3),  # Add BatchNorm2d here
            nn.Tanh(),  # Output in range [-1, 1]
        )
        
    def _get_flattened_size(self):
        x = torch.rand(1, 3, 1, 256, 256)
        x = self.encoder(x)
        return x.view(x.size(0), -1).size(1)
    
    def encode(self, x):
        x = x.permute(0, 2, 1, 3, 4)
        for layer in self.encoder[:1]:
            x = layer(x)                         # After first Conv3d layer
        x = torch.squeeze(x, dim=2)              # Remove the depth dimension (dim=2)   
        for layer in self.encoder[1:]:           # Continue with the Conv2d layers
            x = layer(x)
        x = F.normalize(x, p=2, dim=1)           # L2 normalization
        #x = nn.LayerNorm(x.shape[1:], elementwise_affine=False)(x)
        x = self.flatten(x)    
        
        mu     = self.fc_mu(x)                   # Mean
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar) + 1e-6   # Standard deviation
        eps = torch.randn_like(std)            # Random tensor of the same shape as std
        return mu + eps * std                  # Reparameterization trick

    def decode(self, z):
        #z = (z - z.mean()) / (z.std() + 1e-6)
        #print(f"Input to Decoder: {z.abs().mean().item():.8f}")
        for i, layer in enumerate(self.decoder):
            z = layer(z)
            #if isinstance(layer, (nn.ConvTranspose2d, nn.BatchNorm2d)):
                #print(f"Layer {i} output: {z.abs().mean().item():.8f}")
        output = (z + 1) / 2
        #print(f"Output of Decoder: {output.abs().mean().item():.8f}")
        return output
        
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        z = (z - z.mean()) / (z.std() + 1e-6)
        #z = torch.clamp(z, min=-10, max=10)  # Prevents extreme values
        return self.decode(z), mu, logvar
