import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from torchvision import transforms
from sklearn.metrics import roc_curve, roc_auc_score
from torchmetrics.image import StructuralSimilarityIndexMeasure

from train import train_model
from data_utils import SequenceDataset
from customLoss import PerceptualLoss

# Get train and val sets
def data_train(TRAIN_PATH, transform, batch_size, input_length, val_split=0.2): 
    # Initialize the dataset
    dataset = SequenceDataset(root_dir=TRAIN_PATH, transform=transform, input_length=input_length)
    if val_split == False: val_split = 0       # if no val set is used
    data_loader = dataset.get_dataloader(batch_size=batch_size, shuffle=True)
    train_loader, val_loader = dataset.split_dataloader(data_loader, val_split=0.2)

    print(f"Number of train samples: {len(train_loader.dataset)}")
    print(f"Number of val samples: {len(val_loader.dataset)}")
    return train_loader, val_loader

# Get test set
def data_test(TEST_PATH, transform, batch_size, input_length):
    # Read the .npy file with labels
    list_targets = []
    for f in sorted(os.listdir(TEST_PATH)):
        if f.endswith('.npy'):
            list_targets.append(np.load(TEST_PATH+f)[4:])
    values  = np.concatenate(list_targets)
    targets = [int(x) for x in values]
    
    dataset = SequenceDataset(root_dir=TEST_PATH, transform=transform, input_length=input_length)
    test_loader = dataset.get_dataloader(batch_size=batch_size, shuffle=False)

    print(f"Number of test samples: {len(test_loader.dataset)}")
    return test_loader, targets

# Perceptual Loss for anomaly detection
def test2(test_loader, model, patch_size=(32, 32)):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #using cpu for test
    perceptual_loss_fn = PerceptualLoss(use_L1=True, reduce_batch=True).to(device)

    anomaly_scores = []
    for batch_X, batch_y in test_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        y_out, _, _ = model(batch_X)
        for i in range(len(y_out)):
            reconstructed_img, target_img = (y_out[i]+1)/2, (batch_y[i]+1)/2
            # Ensure the tensors have the same shape
            if reconstructed_img.shape != target_img.shape:
                raise ValueError(f"Reconstructed and target tensors must have the same shape. Got {reconstructed_img.shape} and {target_img.shape} instead.")

            # Get image dimensions
            height, width = reconstructed_img.shape[1], reconstructed_img.shape[2]

            highest_loss = 0
            for i in range(0, height, patch_size[0]):
                for j in range(0, width, patch_size[1]):
                    patch_reconstructed = reconstructed_img[i:i + patch_size[0], j:j + patch_size[1]]
                    patch_target        = target_img[i:i + patch_size[0], j:j + patch_size[1]]

                    # Skip if the patch is smaller than the patch size (e.g., at the edges)
                    if patch_reconstructed.shape != patch_size or patch_target.shape != patch_size:
                        continue
                    
                    # Compute loss for the patch
                    percep_loss  = perceptual_loss_fn(patch_reconstructed, patch_target)
                    
                    if percep_loss > highest_loss: highest_loss = percep_loss # saving the highest loss
            anomaly_scores.append(highest_loss)
    return anomaly_scores

# SSIM for anomaly metric        
#def test(test_loader, model):
#    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#    ssim   = StructuralSimilarityIndexMeasure(data_range=1.0, reduction='none')

#    cont_batch = 0
#    y_pred, anomaly_scores = [], []
#    for batch_X, batch_y in test_loader:
#        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
#        y_out, _, _  = model(batch_X)
#        ssim_scores  = ssim(y_out, (batch_y+1)/2)
        
#        for i in range(len(ssim_scores)):
#            anomaly_scores.append(1 - ssim_scores[i].item())
#    return anomaly_scores


def results(y_true, anomaly_scores, video):
    # Compute ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(y_true, anomaly_scores)
    roc_auc = roc_auc_score(y_true, anomaly_scores)

    print(f"AUC: {roc_auc:.4f}")
    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    # Save the figure
    plt.savefig('new_convAEAD/'+video+'/roc_curve.png', dpi=300, bbox_inches='tight')
    # Close the figure to free memory
    plt.close()

def main():
    IMAGE_SIZE = (128, 128)
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    batch_size   = 8
    input_length = 4
    val_split    = 0.2

    for video in ["Seagull-RGB"]: 
        PATH = "datasets-tests/Drone-Seagull/"+video+"/"
        train_loader, val_loader = data_train(TRAIN_PATH = PATH+"train/", transform = transform, 
                                              batch_size = batch_size, input_length = input_length, val_split=val_split)
        model = train_model(train_loader, val_loader)
        #model, threshold1 = train_model(train_loader)
        #print(f"Threshold 1: {threshold1:.4f}")
    
        test_loader, targets  = data_test(TEST_PATH = PATH+"test/", transform = transform, 
                                          batch_size = batch_size, input_length = input_length)
        
        y_pred = test2(test_loader, model, patch_size=(8, 8))
        results(targets, y_pred, video)

if __name__ == '__main__':
    main()
