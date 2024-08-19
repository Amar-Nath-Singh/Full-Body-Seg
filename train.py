from torch.utils.data import random_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torch.utils.tensorboard import SummaryWriter
from unet import UNet, UNetWithFNN, MinimalUNetWithFNN, ResUNet
from dataset import HumanImageDataset, Transform
from tqdm import tqdm

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, outputs, targets):
        outputs = outputs.sigmoid()
        outputs = (outputs > 0.5).float()
        
        intersection = torch.sum(outputs * targets)
        union = torch.sum(outputs) + torch.sum(targets)
        
        dice = 2.0 * (intersection + self.smooth) / (union + self.smooth)
        return 1 - dice

class IoULoss(nn.Module):
    def __init__(self):
        super(IoULoss, self).__init__()

    def forward(self, outputs, targets):
        outputs = outputs.sigmoid()
        outputs = (outputs > 0.5).float()

        intersection = torch.sum(outputs * targets)
        union = torch.sum(outputs) + torch.sum(targets) - intersection

        iou = (intersection + 1e-6) / (union + 1e-6)
        return 1 - iou

def pixel_wise_loss(outputs, targets):
    criterion = nn.BCEWithLogitsLoss()  # Binary Cross-Entropy with logits
    return criterion(outputs, targets)

def evaluate_model(model, dataloader, device, epoch):
    model.eval()  # Set model to evaluation mode
    running_loss = 0.0

    dice_loss_fn = DiceLoss()
    iou_loss_fn = IoULoss()

    with torch.no_grad():  # No need to track gradients during evaluation
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)

            outputs = model(images)
            loss_pixel = pixel_wise_loss(outputs, masks)
            loss_dice = dice_loss_fn(outputs, masks)
            loss_iou = iou_loss_fn(outputs, masks)

            total_loss = loss_pixel + loss_dice + loss_iou
            running_loss += total_loss.item()

        if epoch == 0:
            writer.add_images('/val/images', images, epoch)
            writer.add_images('/val/masks', masks, epoch)

        writer.add_images('/val/outputs', outputs, epoch)
            
    epoch_loss = running_loss / len(dataloader)
    return epoch_loss

def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    running_loss = 0.0

    dice_loss_fn = DiceLoss()
    iou_loss_fn = IoULoss()

    loop = tqdm(dataloader)
    for images, masks in loop:
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss_pixel = pixel_wise_loss(outputs, masks)
        loss_dice = dice_loss_fn(outputs, masks)
        loss_iou = iou_loss_fn(outputs, masks)

        total_loss = loss_pixel + loss_dice + loss_iou
        total_loss.backward()
        optimizer.step()

        running_loss += total_loss.item()

    epoch_loss = running_loss / len(dataloader)
    return epoch_loss

def main():
    # Configuration
    data_dir = 'segmentation_full_body_mads_dataset_1192_img/segmentation_full_body_mads_dataset_1192_img'
    df_path = 'df.csv'
    batch_size = 16
    num_epochs = 50
    learning_rate = 0.0003
    train_ratio = 0.95


    # Load dataset
    transform = Transform()
    dataset = HumanImageDataset(data_dir=data_dir, df_path=df_path, transform=transform)
    
    # Split dataset into training and testing sets
    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Create DataLoader for training and testing sets
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    # Model
    model = ResUNet(in_channels=3, out_channels=1)
    model = model.to(device)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss = float('inf')  # Initialize with a large value
    best_model_path = f'models/{model._get_name()}/best_model.pt'

    # Training loop
    for epoch in range(num_epochs):
        epoch_loss = train_one_epoch(model, train_dataloader, optimizer, device)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')
        writer.add_scalar('/train/loss', epoch_loss, epoch)

        # Evaluate on validation data
        val_loss = evaluate_model(model, test_dataloader, device, epoch)
        print(f'Epoch {epoch+1}/{num_epochs}, Val Loss: {val_loss:.4f}')
        writer.add_scalar('/val/loss', val_loss, epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f'Saved new best model with loss {best_val_loss:.4f} at epoch {epoch+1}')
        
        torch.save(model.state_dict(), f'models/{model._get_name()}/last_model.pt')

    print("Training complete.")

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    writer = SummaryWriter()
    main()

    writer.close()
