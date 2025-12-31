import os
import pickle
import pandas as pd
import glob
import numpy as np
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.transforms as T
from PIL import Image

import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import matplotlib.pyplot as plt
from torchvision.models import vgg16, VGG16_Weights
    
#Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Path to your merged dataset pickle (from the previous step - see creating_merged_dataset.py)
data_dir = '/path/to/your/dataset'
merged_pickle = os.path.join(data_dir, "merged_dataset_gpu1.pkl")

#Load the merged DataFrame
with open(merged_pickle, "rb") as f:
    merged_df = pickle.load(f)

from sklearn.model_selection import train_test_split

#Add binary column for "No Finding" presence
merged_df["is_no_finding"] = merged_df["Finding Labels"].apply(lambda x: "No Finding" in x)

#Split: 80% train+val, 20% test (stratified on No Finding)
train_val_df, test_df = train_test_split(
    merged_df,
    test_size=0.2,
    random_state=123,
    stratify=merged_df["is_no_finding"]
)

#From train+val, split into 62.5% train, 37.5% val
train_df, val_df = train_test_split(
    train_val_df,
    test_size=0.375,  # 0.375 of 80% = 30%
    random_state=123,
    stratify=train_val_df["is_no_finding"]
)

#1) Define a transform to keep the image in RGB and convert to a tensor [0..1]
transform = T.Compose([
    T.ToTensor()  # shape will be [3, H, W] for RGB
])

#2) Initialise accumulators for each of the 3 channels
sum_pixels = torch.zeros(3)
sum_sq_pixels = torch.zeros(3)
n_pixels = 0

#3) Loop over each row in merged_df (our 3,000 images)
for _, row in merged_df.iterrows():
    img_path = row["file_path"]
    img = Image.open(img_path).convert("RGB")  # ensure 3-channel
    
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)  # Convert ndarray to PIL Image if needed
    
    assert isinstance(img, Image.Image), f"Expected PIL.Image.Image, got {type(img)}"
    
    if not Path(img_path).is_file():
        print(f"Skipping invalid image path: {img_path}")
        continue  # Skip this one

    try:
        img = Image.open(img_path).convert("RGB")
        assert isinstance(img, Image.Image), f"Expected PIL.Image.Image, got {type(img)}"
    except Exception as e:
        print(f"Failed to open image at {img_path}: {e}")
        continue

    # Convert image to a tensor in [0..1], shape [3, H, W]
    tensor_img = transform(img)  

    # tensor_img has shape (3, H, W). Let's get H and W:
    _, H, W = tensor_img.shape
    n_pixels += (H * W)  # for each channel, we'll have H*W pixels

    # Sum of pixel values per channel -> shape [3]
    sum_pixels += tensor_img.sum(dim=[1, 2])
    # Sum of squared pixel values per channel -> shape [3]
    sum_sq_pixels += (tensor_img ** 2).sum(dim=[1, 2])

# 4) Compute mean and std for each channel
mean = sum_pixels / n_pixels
variance = (sum_sq_pixels / n_pixels) - (mean ** 2)
std = torch.sqrt(variance)

transform_train = T.Compose([
    T.Resize((224, 224)),
    T.ColorJitter(brightness=0.2, contrast=0.2),  
    T.ToTensor(),
    T.Normalize(mean=mean.tolist(), std=std.tolist())
])

transform_val_test = T.Compose([
    T.Resize((224, 224)),  
    T.ToTensor(),
    T.Normalize(mean=mean.tolist(), std=std.tolist())
])

##########
class CXRDataset(Dataset):
    def __init__(self, df, transform=None):
        """
        df: DataFrame with columns ['file_path', 'Finding Labels']
        transform: torchvision transforms for preprocessing
        """
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["file_path"]
        label_str = row["Finding Labels"]

        # Open grayscale image and convert to RGB for VGG input
        image = Image.open(img_path).convert("RGB")

        # Binary label: 1 = any disease, 0 = No Finding
        binary_label = 0.0 if label_str == "No Finding" else 1.0

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(binary_label, dtype=torch.float)

train_dataset = CXRDataset(train_df, transform=transform_train)
val_dataset   = CXRDataset(val_df, transform=transform_val_test)
test_dataset  = CXRDataset(test_df, transform=transform_val_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False)

#####MODEL

import sys
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

# Hyperparameters
num_epochs = 25
learning_rate = 1e-6
weight_decay = 1e-1

#Load the pre-trained VGG16 model
model = models.vgg16(weights=VGG16_Weights.DEFAULT)

#Unfreeze all layers initially
for param in model.parameters():
    param.requires_grad = True
    
#Freezing the first block of the VGG16 model (convolutional layers)
for param in model.features[:5].parameters():  # The first block has 10 layers
    param.requires_grad = False
    
#Convert model.features to a list, add Dropout, and make a new Sequential
model.features = nn.Sequential(
    *list(model.features.children()), 
    nn.Dropout(p=0.5)  # Add dropout here
)
#Replacing final layer with binary output
model.classifier = nn.Sequential(
    nn.Linear(25088, 4096),
    nn.BatchNorm1d(4096),
    nn.ReLU(inplace=True),
    nn.Dropout(p=0.5),            # First dropout

    nn.Linear(4096, 4096),
    nn.BatchNorm1d(4096),
    nn.ReLU(inplace=True),
    nn.Dropout(p=0.5),            # Second dropout 

    nn.Linear(4096, 1)            # Final binary output layer
)

# Move model to GPU(s)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = nn.DataParallel(model)

model = model.to(device)

# Optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,         # Optimizer to adjust the learning rate
    mode='min',        # We're minimizing the validation loss
    factor=0.5,        # Multiply LR by 0.5 when reducing
    patience=2       # Wait for 3 epochs with no improvement before reducing LR
)
criterion = nn.BCEWithLogitsLoss()

from sklearn.metrics import confusion_matrix

# --- Evaluation Function ---
def evaluate_and_plot(model, dataloader, device, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for images, binary_label in dataloader:
            images = images.to(device)
            binary_label = binary_label.to(device).unsqueeze(1)

            outputs = model(images)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).int()

            loss = criterion(outputs, binary_label)
            total_loss += loss.item()
            correct += (preds == binary_label.bool()).sum().item()
            total += binary_label.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(binary_label.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Metrics
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    auc_score = roc_auc_score(all_labels, all_probs)

    # Confusion matrix to compute FPR and FNR
    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()

    # Calculate FPR and FNR
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

    return avg_loss, accuracy, precision, recall, f1, auc_score, fpr, fnr

# Metric lists to store data across epochs
train_losses, train_accuracies = [], []
val_losses, val_accuracies = [], []
val_precisions, val_recalls, val_f1s, val_aucs = [], [], [], []
val_fprs, val_fnrs = [], []
best_val_accuracy = 0.0  # Initialize to the worst possible accuracy (0%)

##### Training loop

early_stop_patience = 5
epochs_no_improve = 0
best_val_loss = float('inf')
best_val_accuracy = 0.0  # If you're tracking this separately

for epoch in range(num_epochs):
    model.train()
    epoch_train_loss = 0.0
    correct_train, total_train = 0, 0

    for images, binary_label in train_loader:
        images = images.to(device)
        binary_label = binary_label.to(device).unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, binary_label)
        loss.backward()
        optimizer.step()

        epoch_train_loss += loss.item()
        preds = torch.sigmoid(outputs) > 0.5
        correct_train += (preds == binary_label.bool()).sum().item()
        total_train += binary_label.size(0)

    # Compute training metrics
    avg_train_loss = epoch_train_loss / len(train_loader)
    train_accuracy = correct_train / total_train

    # --- Validation ---
    val_loss, val_acc, val_prec, val_rec, val_f1, val_auc, val_fpr, val_fnr = evaluate_and_plot(model, val_loader, device, criterion)

    # Append metrics
    train_losses.append(avg_train_loss)
    train_accuracies.append(train_accuracy)
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)
    val_precisions.append(val_prec)
    val_recalls.append(val_rec)
    val_f1s.append(val_f1)
    val_aucs.append(val_auc)
    val_fprs.append(val_fpr)
    val_fnrs.append(val_fnr)

    # Learning rate scheduler
    scheduler.step(val_loss)

    print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}, | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val Recall: {val_rec:.4f}")

    # --- Early Stopping Check (still uses val_loss) ---
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        print(f"No improvement in val_loss for {epochs_no_improve} epoch(s).")
        if epochs_no_improve >= early_stop_patience:
            print("Early stopping triggered.")
            break

    # --- Save Best Model Based on Accuracy ---
    if val_acc > best_val_accuracy:
        best_val_accuracy = val_acc
        torch.save(model.state_dict(), 'best_vggfull3_accuracy.pth')
        print(f"Model saved at epoch {epoch+1} with val_acc: {val_acc:.4f}")

# --- Save final epoch metrics to CSV ---
final_metrics = {
    'final_epoch': num_epochs,
    'final_train_loss': train_losses[-1] if len(train_losses) > 0 else None,
    'final_train_acc': train_accuracies[-1] if len(train_accuracies) > 0 else None,
    'final_val_loss': val_losses[-1] if len(val_losses) > 0 else None,
    'final_val_acc': val_accuracies[-1] if len(val_accuracies) > 0 else None,
    'final_val_precision': val_precisions[-1] if len(val_precisions) > 0 else None,
    'final_val_recall': val_recalls[-1] if len(val_recalls) > 0 else None,
    'final_val_f1': val_f1s[-1] if len(val_f1s) > 0 else None,
    'final_val_auc': val_aucs[-1] if len(val_aucs) > 0 else None,
    'final_val_fpr': val_fprs[-1] if len(val_fprs) > 0 else None,
    'final_val_fnr': val_fnrs[-1] if len(val_fnrs) > 0 else None
}

# Save the metrics to CSV
df = pd.DataFrame([final_metrics])
df.to_csv("vggfull3.csv", index=False)

import matplotlib.ticker as ticker

# Accuracy Plot
plt.figure() 
plt.plot(train_accuracies, label="Train Accuracy")
plt.plot(val_accuracies, label="Val Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy over Epochs")
plt.legend()
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(1))
plt.savefig('/rds/general/user/nsl24/home/ML_Project/Data/accuracy_vggfull3_plt.png')
plt.show()


# Loss Plot
plt.figure() 
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss over Epochs")
plt.legend()
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(1))
plt.savefig('/rds/general/user/nsl24/home/ML_Project/Data/loss_vggfull3_plt.png')
plt.show()

###Predicting on Test Set 
# Set device

# Rebuild the same model architecture
model = models.vgg16(weights=VGG16_Weights.DEFAULT)

# Unfreeze all layers initially
for param in model.parameters():
    param.requires_grad = True

# Freeze first block again (if you had done this originally)
for param in model.features[:5].parameters():
    param.requires_grad = False

# Add dropout after features
model.features = nn.Sequential(
    *list(model.features.children()),
    nn.Dropout(p=0.5)
)

# Rebuild classifier with BatchNorm and Dropout
model.classifier = nn.Sequential(
    nn.Linear(25088, 4096),
    nn.BatchNorm1d(4096),
    nn.ReLU(inplace=True),
    nn.Dropout(0.5),

    nn.Linear(4096, 4096),
    nn.BatchNorm1d(4096),
    nn.ReLU(inplace=True),
    nn.Dropout(0.5),

    nn.Linear(4096, 1)
)

model.load_state_dict(torch.load("best_vggfull3_accuracy.pth"))
model = model.to(device)
model.eval()  

all_preds = []
all_probs = []
all_labels = []

with torch.no_grad():
    for images, binary_label in test_loader:
        images = images.to(device)
        binary_label = binary_label.to(device).unsqueeze(1)

        outputs = model(images)
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).int()

        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(binary_label.cpu().numpy())


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds)
recall = recall_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)
auc = roc_auc_score(all_labels, all_probs)

# Confusion matrix to compute FPR and FNR
cm = confusion_matrix(all_labels, all_preds)
tn, fp, fn, tp = cm.ravel()

# Calculate FPR and FNR
fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

#Confusion Matrix
plt.figure() 
cm = confusion_matrix(all_labels, all_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["No Disease", "Disease"],
            yticklabels=["No Disease", "Disease"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig('/rds/general/user/nsl24/home/ML_Project/Data/confusion_vggfull3.png')
plt.show()

#ROC Curve
plt.figure() 
fpr, tpr, _ = roc_curve(all_labels, all_probs)
plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.savefig('/rds/general/user/nsl24/home/ML_Project/Data/roc_vggfull3.png')
plt.show()

print("Test Accuracy:", accuracy)
print("Test Precision:", precision)
print("Test Recall:", recall)
print("Test F1 Score:", f1)
print("Test AUC:", auc)
print("False Positive Rate (FPR):", fpr)
print("False Negative Rate (FNR):", fnr)
print("Confusion Matrix:\n", cm)

df_metrics = pd.DataFrame({
    'accuracy': [accuracy],
    'precision': [precision],
    'recall': [recall],
    'f1_score': [f1],
    'auc': [auc],
    'fnr': [fnr],
})

df_metrics.to_csv("vggfull3_metrics.csv", index=False)

metrics_dict = {
    'train_loss': train_losses,
    'train_accuracy': train_accuracies,
    'val_loss': val_losses,
    'val_accuracy': val_accuracies,
    'val_precision': val_precisions,
    'val_recall': val_recalls,
    'val_f1': val_f1s,
    'val_auc': val_aucs,
    'val_fnr': val_fnrs
}

# Step 2: Convert to a DataFrame
metrics_df = pd.DataFrame(metrics_dict)

# Step 3: Save to CSV
metrics_df.to_csv('training_metrics_vggfull3.csv', index_label='epoch')

