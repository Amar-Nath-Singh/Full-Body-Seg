import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd

class HumanImageDataset(Dataset):
    def __init__(self, data_dir, df_path, transform=None):

        self.data_dir = data_dir
        self.transform = transform
        

        self.df = pd.read_csv(df_path)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # Load image and label
        image_path = os.path.join(self.data_dir, self.df['images'][idx])
        label_path = os.path.join(self.data_dir, self.df['masks'][idx])
        
        image = Image.open(image_path).convert('RGB')  # Convert image to RGB
        label = Image.open(label_path).convert('L')    # Convert label to grayscale
        
        # Apply transformations if any
        if self.transform:
            image, label = self.transform(image, label)
        
        return image, label

class Transform:
    def __init__(self):
        self.image_transform = transforms.Compose([
            transforms.Resize((256, 256)),    # Resize images to 256x256
            transforms.ToTensor(),            # Convert images to tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet stats
        ])
        
        self.label_transform = transforms.Compose([
            transforms.Resize((256, 256)),    # Resize labels to 256x256
            transforms.ToTensor()             # Convert labels to tensor
        ])
        
    def __call__(self, image, label):
        return self.image_transform(image), self.label_transform(label)

# Example usage
if __name__ == "__main__":
    from torch.utils.data import DataLoader
    
    # Define paths to your image and label directories
    # Instantiate the transformation class
    transform = Transform()
    
    # Instantiate the dataset with the transformation
    dataset = HumanImageDataset(data_dir='segmentation_full_body_mads_dataset_1192_img/segmentation_full_body_mads_dataset_1192_img',
                                df_path='df.csv', transform=transform)
    
    # Create a DataLoader
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # Iterate through the dataset
    for images, labels in dataloader:
        print(images.shape)  # (batch_size, channels, height, width)
        print(labels.shape)  # (batch_size, channels, height, width)
