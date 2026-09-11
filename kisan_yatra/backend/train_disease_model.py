"""
Train an EfficientNet crop/plant disease classification model on local images.
Mirrors train_soil_model.py, but uses ImageFolder (one subfolder per class)
since that's the standard layout for public disease datasets (e.g. PlantVillage
on Kaggle: emmarex/plantdisease).

DATASET LAYOUT EXPECTED (put this under "crop diseases/"):

    crop diseases/
        powdery_mildew/   *.jpg
        early_blight/     *.jpg
        late_blight/      *.jpg
        leaf_spot/        *.jpg
        rust/             *.jpg
        bacterial_wilt/   *.jpg
        anthracnose/      *.jpg
        healthy/          *.jpg

IMPORTANT: the folder names become the class labels the model predicts.
Name them exactly to match the keys in DISEASE_REMEDIES inside
disease_detection.py (powdery_mildew, early_blight, late_blight, leaf_spot,
rust, bacterial_wilt, anthracnose, healthy) so the remedy lookup keeps working
once you swap this model in. If your downloaded dataset uses different names
(PlantVillage folders look like "Tomato___Late_blight", "Apple___healthy"
etc.), rename/regroup the folders into the 8 buckets above first — you don't
need per-crop granularity for this UI, just the disease category.

Run this ONCE to train the model, then point disease_detection.py at the
saved .pth instead of calling Roboflow.
"""

import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, models, datasets
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split

# ============================================================================
# CONFIGURATION
# ============================================================================
DISEASE_DATA_DIR = Path(__file__).resolve().parent / "crop diseases"
MODEL_SAVE_DIR = Path(__file__).resolve().parent / "models"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
EPOCHS = 25
LEARNING_RATE = 0.0005
IMG_SIZE = 224
PRINT_EVERY_N_BATCHES = 50  # progress heartbeat within an epoch

print(f"Using device: {DEVICE}")
print(f"Disease data directory: {DISEASE_DATA_DIR}")

# ============================================================================
# TRANSFORMS
# ============================================================================
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def train_disease_classifier():
    if not DISEASE_DATA_DIR.exists():
        raise FileNotFoundError(
            f"Disease data directory not found: {DISEASE_DATA_DIR}\n"
            "Create it and add one subfolder per class (see module docstring)."
        )

    # ImageFolder infers classes from subfolder names, sorted alphabetically.
    full_dataset = datasets.ImageFolder(DISEASE_DATA_DIR, transform=train_transform)
    disease_classes = full_dataset.classes
    targets = np.array(full_dataset.targets)

    print("\n" + "=" * 60)
    print("DISEASE DATASET SUMMARY")
    print("=" * 60)
    total = 0
    for idx, cls in enumerate(disease_classes):
        count = int((targets == idx).sum())
        total += count
        print(f"{cls:20} : {count:4} images")
    print(f"{'Total':20} : {total:4} images")
    print("=" * 60 + "\n")

    if total < 50:
        print("WARNING: very small dataset — accuracy will likely be poor. "
              "Add more images per class before trusting this model.")

    indices = np.arange(len(full_dataset))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.2, random_state=42, stratify=targets
    )

    train_dataset = Subset(full_dataset, train_idx)

    # Eval subset needs the non-augmented transform, so build a second
    # ImageFolder instance over the same directory rather than mutating
    # the shared one mid-training.
    eval_base = datasets.ImageFolder(DISEASE_DATA_DIR, transform=eval_transform)
    test_dataset = Subset(eval_base, test_idx)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}\n")

    # Pre-trained EfficientNet-B0, same backbone as the soil model.
    model = models.efficientnet_b0(weights="IMAGENET1K_V1")
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, len(disease_classes))
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    best_accuracy = 0.0
    best_state = None

    MODEL_SAVE_DIR.mkdir(exist_ok=True)
    checkpoint_path = MODEL_SAVE_DIR / "disease_efficientnet_b0.pth"
    classes_path = MODEL_SAVE_DIR / "disease_classes.pkl"
    with open(classes_path, "wb") as f:
        pickle.dump(disease_classes, f)  # write classes once, up front

    total_batches = len(train_loader)
    print("Starting training...\n")
    for epoch in range(EPOCHS):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for batch_idx, (images, labels) in enumerate(train_loader, start=1):
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

            if batch_idx % PRINT_EVERY_N_BATCHES == 0 or batch_idx == total_batches:
                running_acc = 100 * train_correct / train_total
                print(f"  epoch {epoch+1} batch {batch_idx}/{total_batches} "
                      f"| running train acc: {running_acc:.2f}%")

        model.eval()
        test_correct, test_total = 0, 0
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

        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, checkpoint_path)  # save immediately, not just at the end
            print(f"  → New best (Test Accuracy: {test_accuracy:.2f}%) — checkpoint saved")

    print(f"\n✓ Training complete! Best Test Accuracy: {best_accuracy:.2f}%\n")
    print(f"✓ Model already saved to: {checkpoint_path} (updated after each improving epoch)")
    print(f"✓ Classes saved to: {classes_path}")

    print("\nDisease classes (index order used by the model):")
    for i, cls in enumerate(disease_classes):
        print(f"  {i}: {cls}")

    print("\n" + "=" * 60)
    print("Next step: update disease_detection.py to load this .pth locally")
    print("instead of calling the Roboflow API.")
    print("=" * 60)


if __name__ == "__main__":
    train_disease_classifier()