import os
import torch
from PIL import Image

from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split

class SequenceDataset(Dataset):
    def __init__(self, root_dir, transform=None, input_length=4):
        self.root_dir = root_dir
        self.transform = transform
        self.input_length = input_length
        self.sequences = []
        items = sorted(os.listdir(root_dir))
        for item in items:
            if os.path.isdir(os.path.join(root_dir, item)):
                self.sequences.append(item)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        def reorder_list(list_imgs):
            order = []
            for i in range(len(list_imgs)):
                m = list_imgs[i].split(".avi_")[-1]
                n = m.split(".")[0]
                order.append(int(n))
        
            idx = sorted(range(len(order)), key=lambda k: order[k])
            new_list = []
            for i in idx:
                new_list.append(list_imgs[i])
            return new_list
            
        sequence_path = os.path.join(self.root_dir, self.sequences[idx])
        frames = reorder_list(os.listdir(sequence_path))
        
        input_sequences = []
        target_sequences = []
        
        for i in range(len(frames) - self.input_length):
            input_images = []
            for j in range(self.input_length):
                frame_path = os.path.join(sequence_path, frames[i + j])
                #print(frame_path)
                image = Image.open(frame_path).convert('RGB')
                if self.transform:
                    image = self.transform(image)
                input_images.append(image)
            
            # Stack the 4 frames for input sequence
            input_sequence = torch.stack(input_images, dim=0)  # Shape: [4, C, H, W]
            input_sequences.append(input_sequence)
            
            # The 5th frame as the target
            target_frame_path = os.path.join(sequence_path, frames[i + self.input_length])
            target_image = Image.open(target_frame_path).convert('RGB')
            if self.transform:
                target_image = self.transform(target_image)
            target_sequences.append(target_image)
        
        # Stack all input sequences and target sequences for this folder
        input_sequences = torch.stack(input_sequences, dim=0)  # Shape: [n-4, 4, C, H, W]
        target_sequences = torch.stack(target_sequences, dim=0)  # Shape: [n-4, C, H, W]
        
        return input_sequences, target_sequences

    def get_concatenated_data(self):
        all_input_sequences = []
        all_target_sequences = []

        print(f"Number of sequences: {len(self)}")
        # Iterate over all sequence folders
        for i in range(len(self)):
            input_sequences, target_sequences = self[i]
            all_input_sequences.append(input_sequences)
            all_target_sequences.append(target_sequences)
        
        # Concatenate all input and target sequences
        concatenated_input  = torch.cat(all_input_sequences, dim=0)  # Shape: [total_samples, 4, C, H, W]
        concatenated_target = torch.cat(all_target_sequences, dim=0)  # Shape: [total_samples, C, H, W]
        return concatenated_input, concatenated_target
    
    def get_dataloader(self, batch_size=32, shuffle=True):
        # Use the get_concatenated_data method to create the complete dataset
        concatenated_input, concatenated_target = self.get_concatenated_data()

        # Create a CustomDataset for the concatenated data
        concatenated_dataset = CustomDataset(concatenated_input, concatenated_target)
        
        # Create and return the DataLoader
        return DataLoader(concatenated_dataset, batch_size=batch_size, shuffle=shuffle)

    def split_dataloader(self, data_loader, val_split=0.2):
        # Extract all the data and labels from the data_loader
        all_data, all_labels = [], []
        for data, labels in data_loader:
            all_data.append(data)
            all_labels.append(labels)

        # Concatenate the data and labels to form a single tensor
        all_data = torch.cat(all_data)
        all_labels = torch.cat(all_labels)
        
        # Determine sizes for training and validation sets
        total_size = len(all_data)
        val_size = int(total_size * val_split)
        train_size = total_size - val_size

        # Create TensorDataset for the full dataset
        full_dataset = TensorDataset(all_data, all_labels)
        # Split the dataset
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
        # Create new DataLoader objects for train and validation sets
        train_loader = DataLoader(train_dataset, batch_size=data_loader.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=data_loader.batch_size, shuffle=False)
    
        return train_loader, val_loader
        
class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
    

