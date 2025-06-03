import os
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.utils as vutils
from torchvision.models.efficientnet import EfficientNet_V2_S_Weights


# -----------------------------
# Configuration & Hyperparameters
# -----------------------------
DATASET_PATH = "CUB_200_2011"
IMAGES_FOLDER = os.path.join(DATASET_PATH, "images")

BB_FILE = os.path.join(DATASET_PATH, "bounding_boxes.txt")
IMAGES_FILE = os.path.join(DATASET_PATH, "images.txt")
LABELS_FILE = os.path.join(DATASET_PATH, "image_class_labels.txt")
SPLIT_FILE = os.path.join(DATASET_PATH, "train_test_split.txt")

IMG_SIZE = 384
BATCH_SIZE = 32
FREEZE_EPOCHS = 6  # Phase 1 epochs
STAGE_TWO = 44  # Phase 2 epochs
STAGE_THREE = 12 # Phase 3 epochs

# -----------------------------
# Custom Dataset
# -----------------------------
class CUBDataset(Dataset):
    def __init__(self, dataframe, images_folder, img_size, training=True):
        self.df = dataframe.reset_index(drop=True)
        self.images_folder = images_folder
        self.img_size = img_size
        self.training = training

        # Define transforms: augmentations for training, basic resize for validation
        if self.training:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(.4),
                transforms.RandomRotation(.25),
                transforms.ColorJitter(brightness=.3,contrast=.23,saturation=.2,hue=.2),
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.RandomErasing(.2),
                transforms.Normalize(mean=[.485, .456, .406], std=[.229, .224, .225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[.485, .456, .406], std=[.229, .224, .225])
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = os.path.join(self.images_folder, row['path'])
        label = int(row['class_label'])
        x, y, w, h = int(row['x']), int(row['y']), int(row['w']), int(row['h'])

        # Open the image and ensure it is RGB
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            width, height = img.size
            # Ensure the bounding box is valid
            x = min(x, width - 1)
            y = min(y, height - 1)
            w = min(w, width - x)
            h = min(h, height - y)
            # Crop: (left, upper, right, lower)
            cropped = img.crop((x, y, x + w, y + h))

        image = self.transform(cropped)
        return image, label

def main():
    # -----------------------------
    # Load and Prepare CSV Data
    # -----------------------------
    images_df = pd.read_csv(IMAGES_FILE, sep=" ", header=None, names=["image_id", "path"])
    labels_df = pd.read_csv(LABELS_FILE, sep=" ", header=None, names=["image_id", "label"])
    split_df = pd.read_csv(SPLIT_FILE, sep=" ", header=None, names=["image_id", "is_train"])
    bboxes_df = pd.read_csv(BB_FILE, sep=" ", header=None, names=["image_id", "x", "y", "w", "h"])

    # Merge into one DataFrame
    df = images_df.merge(labels_df, on="image_id").merge(split_df, on="image_id").merge(bboxes_df, on="image_id")
    df["class_label"] = df["label"] - 1  # Convert to 0-based labels
    unique_labels = sorted(df["label"].unique())
    num_classes = len(unique_labels)

    # Split train/validation
    train_df = df[df["is_train"] == 1]
    val_df = df[df["is_train"] == 0]
    val_df = df[df["is_train"] == 0]

    print(f"Number of classes: {num_classes}")

    # -----------------------------
    # Create datasets and DataLoaders
    # -----------------------------
    train_dataset = CUBDataset(train_df, IMAGES_FOLDER, IMG_SIZE, training=True)
    val_dataset = CUBDataset(val_df, IMAGES_FOLDER, IMG_SIZE, training=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # -----------------------------
    # Build the Model
    # -----------------------------
    model = models.efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.39), # adjust
        nn.Linear(in_features, num_classes)
    )

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Freeze the base feature extractor for phase 1
    for param in model.features.parameters():
        param.requires_grad = False

    # -----------------------------
    # Loss, Optimizer, and Scheduler Setup
    # -----------------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=7e-4, weight_decay=.0005)
    scheduler = ExponentialLR(optimizer, gamma=0.98)

    # -----------------------------
    # TensorBoard Setup
    # -----------------------------
    writer = SummaryWriter(log_dir="runs/experiment1")

    # -----------------------------
    # Training and Validation Functions
    # -----------------------------
    def train_epoch(model, loader, criterion, optimizer, device):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for i, (images, labels) in enumerate(loader):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

            if i % 100 == 0:
                img_grid = vutils.make_grid(images[:16], nrow=4, normalize=True)
                writer.add_image("training images",img_grid)

        return running_loss / total, correct / total

    def validate_epoch(model, loader, criterion, device):
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()
        return running_loss / total, correct / total

    # -----------------------------
    # Phase 1: Training with Frozen Base
    # -----------------------------
    print("\n=====================")
    print("Phase 1: Training with frozen base features")
    for epoch in range(FREEZE_EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Epoch {epoch + 1}/{FREEZE_EPOCHS} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Log metrics to TensorBoard
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Accuracy/Train", train_acc, epoch)
        writer.add_scalar("Loss/Validation", val_loss, epoch)
        writer.add_scalar("Accuracy/Validation", val_acc, epoch)

    # -----------------------------
    # Phase 2: Fine-tuning (Unfreeze last 10 layers of base)
    # -----------------------------
    features_children = list(model.features.children())
    for layer in features_children[-4:]:
        for param in layer.parameters():
            param.requires_grad = True

    optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=.0027)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode= 'min', factor= .1, patience=2, verbose= True)


    print("\n=====================")
    print("Phase 2: Fine-tuning the network")
    for epoch in range(STAGE_TWO):
        global_epoch = FREEZE_EPOCHS + epoch
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        scheduler.step(val_loss )

        print(f"Epoch {global_epoch + 1}/{FREEZE_EPOCHS + STAGE_TWO + STAGE_THREE} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Log metrics to TensorBoard
        writer.add_scalar("Loss/Train", train_loss, global_epoch)
        writer.add_scalar("Accuracy/Train", train_acc, global_epoch)
        writer.add_scalar("Loss/Validation", val_loss, global_epoch)
        writer.add_scalar("Accuracy/Validation", val_acc, global_epoch)

    #final stage 3

    features_children = list(model.features.children())
    for layer in features_children[-6:]:
        for param in layer.parameters():
            param.requires_grad = True

    optimizer = optim.Adam(model.parameters(), lr=1e-6, weight_decay=.003)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode= 'min', factor= .15, patience=2, verbose= True)



    print("\n=====================")
    print("Phase 3: ultra fine-tuning the network")
    for epoch in range(STAGE_THREE):
        global_epoch = FREEZE_EPOCHS + STAGE_TWO + epoch
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        print(
            f"Epoch {global_epoch + 1}/{FREEZE_EPOCHS + STAGE_TWO + STAGE_THREE} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Log metrics to TensorBoard
        writer.add_scalar("Loss/Train", train_loss, global_epoch)
        writer.add_scalar("Accuracy/Train", train_acc, global_epoch)
        writer.add_scalar("Loss/Validation", val_loss, global_epoch)
        writer.add_scalar("Accuracy/Validation", val_acc, global_epoch)

    # -----------------------------
    # Final Evaluation and Saving
    # -----------------------------
    val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
    print("\nFinal Evaluation on Validation Set")
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

    torch.save(model.state_dict(), "bird_model.pth")
    print("Model saved as bird_model.pth")

    writer.close()


if __name__ == '__main__':
    main()
