"""
Train the EfficientNet soil classification model on local soil images.
This script organizes soil images by type and fine-tunes the model.
Run this ONCE to train the model, then restart the Streamlit app.
"""

import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split

# ============================================================================
# CONFIGURATION
# ============================================================================
SOIL_DATA_DIR = Path(__file__).resolve().parent / "soil data pics"
MODEL_SAVE_DIR = Path(__file__).resolve().parent / "models"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8
EPOCHS = 50
LEARNING_RATE = 0.0005
IMG_SIZE = 224

print(f"Using device: {DEVICE}")
print(f"Soil data directory: {SOIL_DATA_DIR}")

# ============================================================================
# DATASET PREPARATION
# ============================================================================

def get_soil_class_from_filename(filename):
    """Extract soil class from filename."""
    filename_lower = filename.lower()
    
    # Define soil type keywords and their standardized names
    soil_types = {
        'alluvial': 'Alluvial soil',
        'black': 'Black Soil',
        'clay': 'Clay soil',
        'laterite': 'Laterite soil',
        'micaceous': 'Micaceous soil',
        'red': 'Red soil',
        'sandy': 'Sandy soil'
    }
    
    for keyword, soil_class in soil_types.items():
        if keyword in filename_lower:
            return soil_class
    
    return None

def convert_avif_images():
    """Convert AVIF images to PNG format."""
    from PIL import Image
    converted = []
    for image_file in SOIL_DATA_DIR.iterdir():
        if image_file.suffix.lower() == '.avif':
            try:
                img = Image.open(image_file)
                png_path = image_file.with_suffix('.png')
                img.convert('RGB').save(png_path)
                converted.append((image_file.name, png_path.name))
                print(f"Converted: {image_file.name} → {png_path.name}")
            except Exception as e:
                print(f"Error converting {image_file.name}: {e}")
    return converted

def organize_soil_images():
    """Organize soil images by class from filenames."""
    soil_images = {}
    
    if not SOIL_DATA_DIR.exists():
        raise FileNotFoundError(f"Soil data directory not found: {SOIL_DATA_DIR}")
    
    # Supported image formats
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.avif'}
    
    for image_file in SOIL_DATA_DIR.iterdir():
        if image_file.suffix.lower() not in image_extensions:
            continue
        
        soil_class = get_soil_class_from_filename(image_file.name)
        if soil_class is None:
            print(f"Warning: Could not classify image: {image_file.name}")
            continue
        
        if soil_class not in soil_images:
            soil_images[soil_class] = []
        
        soil_images[soil_class].append(image_file)
    
    # Print dataset summary
    print("\n" + "="*60)
    print("SOIL DATASET SUMMARY")
    print("="*60)
    total_images = 0
    for soil_class, images in sorted(soil_images.items()):
        count = len(images)
        total_images += count
        print(f"{soil_class:20} : {count:3} images")
    print(f"{'Total':20} : {total_images:3} images")
    print("="*60 + "\n")
    
    return soil_images

# ============================================================================
# PYTORCH DATASET
# ============================================================================

class SoilImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            # Convert WEBP and AVIF to RGB
            image = Image.open(img_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            
            return image, label
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a black image as fallback
            return torch.zeros(3, self.transform.transforms[0].size[0], self.transform.transforms[0].size[1]), label

# ============================================================================
# MODEL TRAINING
# ============================================================================

def train_soil_classifier():
    """Train EfficientNet model on soil images."""
    
    # Convert AVIF images to PNG first
    print("Converting AVIF images to PNG...\n")
    convert_avif_images()
    
    # Organize images by soil type
    soil_images = organize_soil_images()
    soil_classes = sorted(soil_images.keys())
    class_to_idx = {cls: idx for idx, cls in enumerate(soil_classes)}
    
    # Flatten image paths and labels
    all_image_paths = []
    all_labels = []
    
    for soil_class, image_paths in soil_images.items():
        for img_path in image_paths:
            all_image_paths.append(img_path)
            all_labels.append(class_to_idx[soil_class])
    
    # Split into train and test sets (80-20)
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        all_image_paths, all_labels, test_size=0.2, random_state=42, stratify=all_labels
    )
    
    # Data transforms with stronger augmentation
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(25),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Create datasets and dataloaders
    train_dataset = SoilImageDataset(train_paths, train_labels, transform)
    test_dataset = SoilImageDataset(test_paths, test_labels, test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}\n")
    
    # Load pre-trained EfficientNet model
    model = models.efficientnet_b0(weights='IMAGENET1K_V1')
    
    # Replace classifier for soil classes
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, len(soil_classes))
    model = model.to(DEVICE)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    
    best_accuracy = 0.0
    
    # Training loop
    print("Starting training...\n")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        # Validation
        model.eval()
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
        
        train_accuracy = 100 * train_correct / train_total
        test_accuracy = 100 * test_correct / test_total
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Loss: {train_loss/len(train_loader):.4f} | "
              f"Train Acc: {train_accuracy:.2f}% | Test Acc: {test_accuracy:.2f}%")
        
        scheduler.step(test_accuracy)
        
        # Save best model
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            print(f"  → Saving best model (Test Accuracy: {test_accuracy:.2f}%)")
    
    print(f"\n✓ Training complete! Best Test Accuracy: {best_accuracy:.2f}%\n")
    
    # Save model and class mapping
    MODEL_SAVE_DIR.mkdir(exist_ok=True)
    
    model_path = MODEL_SAVE_DIR / "efficientnet_b0_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"✓ Model saved to: {model_path}")
    
    classes_path = MODEL_SAVE_DIR / "soil_classes.pkl"
    with open(classes_path, 'wb') as f:
        pickle.dump(soil_classes, f)
    print(f"✓ Classes saved to: {classes_path}")
    
    print("\nSoil classes:")
    for i, cls in enumerate(soil_classes):
        print(f"  {i}: {cls}")
    
    print("\n" + "="*60)
    print("Training complete! Restart the Streamlit app to use the new model.")
    print("="*60)

if __name__ == "__main__":
    train_soil_classifier()
