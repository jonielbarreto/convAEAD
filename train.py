import torch
import statistics
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from customLoss import PerceptualLoss
from convLSTM_AEModel import VariationalAutoencoder
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics.image import StructuralSimilarityIndexMeasure

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0, verbose=False):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def train_model(train_loader, val_loader=None):
    model = VariationalAutoencoder()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-6)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5, verbose=True)

    #criterion = nn.L1Loss()  # Use L1Loss instead of MSELoss
    def initialize_weights(m):
        if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu', a=0.3)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    model.apply(initialize_weights)
    
    # Perceptual Loss
    perceptual_loss_fn = PerceptualLoss(use_L1=True, reduce_batch=True).to(device)

    #def CombinedLoss(reconstructed_y, y, mu, logvar, kl_weight=0.01, alpha=0.5, betha=0.5):
    #    reconstructed_y, y = reconstructed_y, (y + 1) / 2
    #    ssim = StructuralSimilarityIndexMeasure(data_range=(0, 1.0), reduction='none')
    #    ssim_score = ssim(reconstructed_y, y)
    #    ssim_loss  = 1 - ssim_score.mean()
    #    reconstruction_loss  = F.mse_loss(reconstructed_y, y, reduction='mean')
    #    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    #    total_loss = alpha * ssim_loss + betha * reconstruction_loss + kl_weight * kl_divergence
    #    return total_loss

    num_epochs = 500
    early_stopping = EarlyStopping(patience=10, min_delta=0.0001, verbose=True)

    for epoch in range(num_epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            output, mu, logvar = model(batch_X)
            #print(f"Output shape in training loop: {output.shape}")  # Should be [8, 3, 128, 128]
            #loss = CombinedLoss(output, batch_y, mu, logvar, kl_weight=0.01, alpha=0.5, betha=0.5)
            #print(f"Input to PerceptualLoss: {output.shape}, {((batch_y + 1) / 2).shape}")  # Both should be [8, 3, 128, 128]
            loss = perceptual_loss_fn(output, (batch_y+1)/2) #(batch_y+1)/2 to make the output range [0,1]
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            
        #for name, param in model.decoder.named_parameters():
            #if param.grad is not None:
                #print(f"{name} gradient mean: {param.grad.abs().mean().item():.8f}")
                
        if val_loader:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for val_X, val_y in val_loader:
                    val_X, val_y = val_X.to(device), val_y.to(device)
                    val_output, mu, logvar = model(val_X)
                    #val_loss += CombinedLoss(val_output, val_y, mu, logvar, kl_weight=0.01, alpha=0.5, betha=0.5)
                    val_loss += perceptual_loss_fn(val_output, (val_y+1)/2)

            val_loss = val_loss / len(val_loader)
            scheduler.step(val_loss)
            
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}')
            early_stopping(val_loss)
            if early_stopping.early_stop:
                print("Early stopping")
                break
        else:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    return model
    
"""
    similarity = []
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0, reduction='none')
    for batch_X, batch_y in train_loader:
        output, _, _ = model(batch_X.to(device))
        ssim_score   = ssim(output, batch_y.to(device))
        similarity.append(ssim_score.mean().item())
    thr1 = statistics.mean(similarity)
    return model, thr1
"""
