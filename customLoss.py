import torch
import torch.nn as nn
import torchvision.models as models

class PerceptualLoss(nn.Module):
    def __init__(self, layers=[4, 9, 16], use_L1=True, reduce_batch=True):
        """
        Args:
            layers: List of indices of VGG16 layers to extract features from.
            use_L1: If True, uses L1 loss; otherwise, uses L2 loss.
            reduce_batch: If True, returns a scalar loss (averaged over batch).
                          If False, returns a tensor of shape [B] (per-sample loss).
        """
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg16(pretrained=True).features  # Load pre-trained VGG16
        self.selected_layers = layers
        self.vgg_layers = nn.Sequential(*list(vgg.children())[:max(layers) + 1])  # Keep up to the max layer used
        for param in self.vgg_layers.parameters():
            param.requires_grad = False  # Freeze VGG weights

        self.criterion = nn.L1Loss(reduction='none') if use_L1 else nn.MSELoss(reduction='none')
        self.reduce_batch = reduce_batch  # Flag to decide output format

    def forward(self, pred, target):
        """
        Compute Perceptual Loss.
        
        Args:
            pred: Predicted frame (B, C, H, W).
            target: Ground-truth frame (B, C, H, W).
        Returns:
            - If reduce_batch=True: returns a scalar loss.
            - If reduce_batch=False: returns a tensor of shape [B] (one loss per sample).
        """
        pred_features = self.extract_features(pred)
        target_features = self.extract_features(target)
        
        losses = []
        for pf, tf in zip(pred_features, target_features):
            layer_loss = self.criterion(pf, tf)  # Loss shape: [B, C, H, W]
            layer_loss = layer_loss.mean(dim=[1, 2, 3])  # Reduce over C, H, W (keeping batch)
            losses.append(layer_loss)  # List of shape [B] losses per layer

        loss = sum(losses)  # Sum across selected layers → Shape: [B]
        
        if self.reduce_batch:
            return loss.mean()  # Return scalar (averaged over batch)
        else:
            return loss  # Return per-sample loss (shape: [B])

    def extract_features(self, x):
        """Extract features from selected layers of VGG."""
        features = []
        for i, layer in enumerate(self.vgg_layers):
            x = layer(x)
            if i in self.selected_layers:
                features.append(x)
        return features

