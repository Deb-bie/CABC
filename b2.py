import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.model_selection import train_test_split
import os
import cv2
import io
from tqdm import tqdm  # For progress bars

# Constants
data_path = "../../../data/BreaKHis_Total_dataset"
labels = ['benign', 'malignant']

# Improved model parameters
d_model = 128  # Increased from 12 to 128 for more representational power
n_classes = 2
img_size = 224  # Increased from 32 to 64 for better resolution
patch_size = 8  # Reduced from 16 to 8 to have more patches per image
n_channels = 1
n_heads = 8  # Increased from 3 to 8
n_layers = 6  # Increased from 3 to 6
dropout_rate = 0.3  # Added dropout rate for regularization
batch_size = 32  # Reduced batch size due to larger model
epochs = 20  # Increased from 5 to 20
alpha = 0.001  # Adjusted learning rate
weight_decay = 1e-4  # Added weight decay for regularization



# Patch Embeddings
class PatchEmbedding(nn.Module):
    def __init__(self, d_model, img_size, patch_size, n_channels):
        super().__init__()

        self.d_model = d_model
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_channels = n_channels

        # Convert scalar img_size/patch_size to tuples if they're not already
        self.img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        self.patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size

        self.linear_project = nn.Conv2d(
            self.n_channels, 
            self.d_model, 
            kernel_size=self.patch_size, 
            stride=self.patch_size
        )

    def forward(self, x):
        x = self.linear_project(x)  # (B, C, H, W) -> (B, d_model, P_col, P_row)
        x = x.flatten(2)  # (B, d_model, P_col, P_row) -> (B, d_model, P)
        x = x.transpose(-2, -1)  # (B, d_model, P) -> (B, P, d_model)
        return x

# Class Token and Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length, dropout_rate=0.1):
        super().__init__()

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.dropout = nn.Dropout(dropout_rate)  # Added dropout

        # Creating positional encoding
        pe = torch.zeros(max_seq_length, d_model)

        for pos in range(max_seq_length):
            for i in range(d_model):
                if i % 2 == 0:
                    pe[pos][i] = np.sin(pos/(10000 ** (i/d_model)))
                else:
                    pe[pos][i] = np.cos(pos/(10000 ** ((i-1)/d_model)))

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # Expand to have class token for every image in batch
        tokens_batch = self.cls_token.expand(x.size()[0], -1, -1)

        # Adding class tokens to the beginning of each embedding
        x = torch.cat((tokens_batch, x), dim=1)

        # Add positional encoding to embeddings
        x = x + self.pe[:, :x.size(1), :]
        
        # Apply dropout
        x = self.dropout(x)

        return x

# Attention Head
class AttentionHead(nn.Module):
    def __init__(self, d_model, head_size, dropout_rate=0.1):
        super().__init__()
        self.head_size = head_size

        self.query = nn.Linear(d_model, head_size)
        self.key = nn.Linear(d_model, head_size)
        self.value = nn.Linear(d_model, head_size)
        self.dropout = nn.Dropout(dropout_rate)  # Added dropout

    def forward(self, x):
        # Obtaining Queries, Keys, and Values
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # Dot Product of Queries and Keys
        attention = Q @ K.transpose(-2, -1)

        # Scaling
        attention = attention / (self.head_size ** 0.5)

        attention = torch.softmax(attention, dim=-1)
        
        # Apply dropout to attention weights
        attention = self.dropout(attention)

        attention = attention @ V

        return attention

# Multi-head Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout_rate=0.1):
        super().__init__()
        self.head_size = d_model // n_heads

        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout_rate)  # Added dropout

        self.heads = nn.ModuleList([AttentionHead(d_model, self.head_size, dropout_rate) for _ in range(n_heads)])

    def forward(self, x):
        # Combine attention heads
        out = torch.cat([head(x) for head in self.heads], dim=-1)

        out = self.W_o(out)
        out = self.dropout(out)  # Apply dropout

        return out

# Transformer Encoder
class TransformerEncoder(nn.Module):
    def __init__(self, d_model, n_heads, dropout_rate=0.1, r_mlp=4):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        # Sub-Layer 1 Normalization
        self.ln1 = nn.LayerNorm(d_model)

        # Multi-Head Attention
        self.mha = MultiHeadAttention(d_model, n_heads, dropout_rate)

        # Sub-Layer 2 Normalization
        self.ln2 = nn.LayerNorm(d_model)

        # Multilayer Perception
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model*r_mlp),
            nn.GELU(),
            nn.Dropout(dropout_rate),  # Added dropout
            nn.Linear(d_model*r_mlp, d_model),
            nn.Dropout(dropout_rate)   # Added dropout
        )

    def forward(self, x):
        # Residual Connection After Sub-Layer 1
        out = x + self.mha(self.ln1(x))

        # Residual Connection After Sub-Layer 2
        out = out + self.mlp(self.ln2(out))

        return out

# Vision Transformer Model
class VisionTransformer(nn.Module):
    def __init__(self, d_model, img_size, patch_size, n_channels, n_heads, n_layers, dropout_rate=0.1):
        super().__init__()

        # Convert scalar img_size/patch_size to tuples if they're not already
        self.img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        self.patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        
        assert self.img_size[0] % self.patch_size[0] == 0 and self.img_size[1] % self.patch_size[1] == 0, \
            "img_size dimensions must be divisible by patch_size dimensions"
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_channels = n_channels
        self.n_heads = n_heads
        self.dropout_rate = dropout_rate

        self.n_patches = (self.img_size[0] * self.img_size[1]) // (self.patch_size[0] * self.patch_size[1])
        self.max_seq_length = self.n_patches + 1

        self.patch_embedding = PatchEmbedding(self.d_model, self.img_size, self.patch_size, self.n_channels)
        self.positional_encoding = PositionalEncoding(self.d_model, self.max_seq_length, dropout_rate)
        self.transformer_encoder = nn.Sequential(*[
            TransformerEncoder(self.d_model, self.n_heads, dropout_rate) for _ in range(n_layers)
        ])

        # Classification MLP with single output for binary classification
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.d_model // 2, 1)  # Single output for binary classification
            # Note: No sigmoid here as we'll use BCEWithLogitsLoss
        )

    def forward(self, images):
        x = self.patch_embedding(images)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)
        x = self.classifier(x[:, 0])  # Only use the class token
        return x

# # Custom Dataset with Augmentation
# class MedicalImageDataset(Dataset):
#     def __init__(self, images, labels, transform=None, augment=False):
#         self.images = images
#         self.labels = labels
#         self.transform = transform
#         self.augment = augment
        
#         # Data augmentation transforms
#         if augment:
#             self.augmentation = T.Compose([
#                 T.RandomHorizontalFlip(p=0.5),
#                 T.RandomVerticalFlip(p=0.5),
#                 T.RandomRotation(degrees=15),
#                 T.RandomAffine(degrees=0, translate=(0.1, 0.1)),
#                 T.ColorJitter(brightness=0.2, contrast=0.2)
#             ])
        
#     def __len__(self):
#         return len(self.images)
    
#     def __getitem__(self, idx):
#         image = self.images[idx]
#         label = self.labels[idx]
        
#         # Convert numpy array to tensor and add channel dimension
#         image = torch.from_numpy(image).float().unsqueeze(0)
        
#         # Apply data augmentation if enabled
#         if self.augment:
#             image = self.augmentation(image)
            
#         # Apply normalization transform
#         if self.transform:
#             image = self.transform(image)
            
#         return image, label




# Custom Dataset with Class-Specific Augmentation
class MedicalImageDataset(Dataset):
    def __init__(self, images, labels, transform=None, augment=False, 
                 benign_augment_factor=3):  # New parameter for benign augmentation factor
        self.images = images
        self.labels = labels
        self.transform = transform
        self.augment = augment
        self.benign_augment_factor = benign_augment_factor
        
        # Store indices by class for controlled augmentation
        self.benign_indices = [i for i, label in enumerate(labels) if label == 0]
        self.malignant_indices = [i for i, label in enumerate(labels) if label == 1]
        
        print(f"Dataset contains {len(self.benign_indices)} benign and {len(self.malignant_indices)} malignant samples")
        
        # Standard data augmentation transforms
        if augment:
            self.standard_augmentation = T.Compose([
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.5),
                T.RandomRotation(degrees=15),
                T.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                T.ColorJitter(brightness=0.2, contrast=0.2)
            ])
            
            # More aggressive augmentation for benign class
            self.aggressive_augmentation = T.Compose([
                T.RandomHorizontalFlip(p=0.7),
                T.RandomVerticalFlip(p=0.7),
                T.RandomRotation(degrees=30),
                T.RandomAffine(degrees=15, translate=(0.2, 0.2), scale=(0.8, 1.2), shear=15),
                T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1)
            ])
    
    def __len__(self):
        if not self.augment:
            return len(self.images)
        
        # When augmenting, we're essentially expanding the benign class
        return len(self.malignant_indices) + len(self.benign_indices) * self.benign_augment_factor
    
    def __getitem__(self, idx):
        if not self.augment:
            # Regular behavior for validation/test sets
            image = self.images[idx]
            label = self.labels[idx]
            
            # Convert numpy array to tensor and add channel dimension
            image = torch.from_numpy(image).float().unsqueeze(0)
            
            # Apply normalization transform
            if self.transform:
                image = self.transform(image)
            
            # For binary cross-entropy: convert label to float tensor
            label = torch.tensor([float(label)], dtype=torch.float32)
            
            return image, label
        
        # Training set with augmentation
        if idx < len(self.malignant_indices):
            # Return a malignant sample
            real_idx = self.malignant_indices[idx]
            image = self.images[real_idx]
            label = 1.0  # Malignant
            
            # Convert numpy array to tensor and add channel dimension
            image = torch.from_numpy(image).float().unsqueeze(0)
            
            # Apply standard augmentation
            image = self.standard_augmentation(image)
        else:
            # Return a benign sample with more aggressive augmentation
            # Map the index to one of the benign indices (with wrapping)
            adjusted_idx = (idx - len(self.malignant_indices)) % len(self.benign_indices)
            real_idx = self.benign_indices[adjusted_idx]
            image = self.images[real_idx]
            label = 0.0  # Benign
            
            # Convert numpy array to tensor and add channel dimension
            image = torch.from_numpy(image).float().unsqueeze(0)
            
            # Randomly choose between standard and aggressive augmentation
            # With higher probability for aggressive on benign samples
            if np.random.random() < 0.7:  # 70% chance of aggressive augmentation
                image = self.aggressive_augmentation(image)
            else:
                image = self.standard_augmentation(image)
        
        # Apply normalization transform
        if self.transform:
            image = self.transform(image)
        
        # For binary cross-entropy: convert label to float tensor
        label = torch.tensor([float(label)], dtype=torch.float32)
        
        return image, label

# Data loading functions
def loading_data(data_dir, img_size):
    data = []
    labels_list = []

    for label in labels:
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        
        try:
            files = os.listdir(path)
        except FileNotFoundError:
            print(f"Warning: Directory {path} not found. Skipping.")
            continue
            
        total_files = len(files)

        print(f"Loading {label} images ({total_files} files)")

        for i, img in enumerate(tqdm(files, desc=f"Loading {label}")):
            img_path = os.path.join(path, img)
            img_arr = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img_arr is not None:
                resized_arr = cv2.resize(img_arr, (img_size, img_size))
                data.append(resized_arr)
                labels_list.append(class_num)
            else:
                print(f"Warning: Unable to read image {img_path}")

    return np.array(data), np.array(labels_list)

def preprocess_data(data, labels):
    # Enhanced preprocessing with standardization
    X_data = np.array(data).astype('float32')
    X_data = (X_data - X_data.mean()) / (X_data.std() + 1e-7)  # Adding epsilon to avoid division by zero
    print(f"Data shape after preprocessing: {X_data.shape}")
    
    y_data = np.array(labels)
    
    # Print class distribution
    unique, counts = np.unique(y_data, return_counts=True)
    class_distribution = dict(zip(unique, counts))
    for class_idx, count in class_distribution.items():
        print(f"Class {labels[class_idx]}: {count} samples")
    
    return X_data, y_data

# Training function
def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in tqdm(dataloader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # Calculate accuracy during training - binary classification
        predicted = (torch.sigmoid(outputs) > 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc


# Validation function
def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # For per-class accuracy
    class_correct = [0, 0]  # benign, malignant
    class_total = [0, 0]
    
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Validating"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            
            # Calculate accuracy - binary classification
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            
            # Store predictions and true labels for later analysis
            predictions.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
            
            # Overall accuracy
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Per-class accuracy
            for i in range(len(labels)):
                label = int(labels[i].item())
                class_total[label] += 1
                if predicted[i].item() == label:
                    class_correct[label] += 1
    
    val_loss = running_loss / len(dataloader)
    val_acc = 100 * correct / total
    
    # Per-class accuracy
    benign_acc = 100 * class_correct[0] / class_total[0] if class_total[0] > 0 else 0
    malignant_acc = 100 * class_correct[1] / class_total[1] if class_total[1] > 0 else 0
    
    # Calculate additional metrics if this is test evaluation
    if len(dataloader) < 10:  # Assuming test set is smaller than validation set
        from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
        
        report = classification_report(true_labels, predictions, target_names=['Benign', 'Malignant'])
        conf_matrix = confusion_matrix(true_labels, predictions)
        
        # Try to calculate AUC (may fail if only one class is present)
        try:
            auc = roc_auc_score([l[0] for l in true_labels], [p[0] for p in predictions])
        except Exception as e:
            auc = None
            print(f"Could not calculate AUC: {e}")
        
        return val_loss, val_acc, benign_acc, malignant_acc, report, conf_matrix, auc
    
    return val_loss, val_acc, benign_acc, malignant_acc

# Main execution
if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device, f"({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else "")

    # Define normalization transform
    transform = T.Compose([
        T.Normalize((0.5,), (0.5,))
    ])

    try:
        print(f"Loading data from {data_path}...")
        data, labels_data = loading_data(data_path, img_size)
        X, y = preprocess_data(data, labels_data)
        
        # Split data into train/validation/test
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, 
            y, 
            test_size=0.3,
            random_state=42,
            stratify=y
        )
        
        # Further split temp into validation and test
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp,
            y_temp,
            test_size=0.5,  # 50% of 30% = 15% of original data
            random_state=42,
            stratify=y_temp
        )
        
        print(f"Train set: {len(X_train)} images")
        print(f"Validation set: {len(X_val)} images")
        print(f"Test set: {len(X_test)} images")
        
        # Create datasets with class-specific augmentation for training
        # Oversampling benign class by a factor of 3
        benign_augment_factor = 3
        train_dataset = MedicalImageDataset(X_train, y_train, transform, augment=True, 
                                           benign_augment_factor=benign_augment_factor)
        val_dataset = MedicalImageDataset(X_val, y_val, transform, augment=False)
        test_dataset = MedicalImageDataset(X_test, y_test, transform, augment=False)
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=2)
        val_loader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size, num_workers=2)
        test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size, num_workers=2)
        
        # Initialize model
        transformer = VisionTransformer(
            d_model, img_size, patch_size, n_channels, 
            n_heads, n_layers, dropout_rate
        ).to(device)
        
        # Binary Cross-Entropy Loss with Logits
        # Add class weights if the dataset is imbalanced
        pos_weight = None
        if hasattr(train_dataset, 'benign_indices') and hasattr(train_dataset, 'malignant_indices'):
            # Calculate weights inversely proportional to class frequencies
            n_benign = len(train_dataset.benign_indices)
            n_malignant = len(train_dataset.malignant_indices)
            if n_benign > 0 and n_malignant > 0:
                pos_weight = torch.tensor([n_benign / n_malignant]).to(device)
                print(f"Using positive class weight: {pos_weight.item()}")
        
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        # Optimizer with weight decay
        optimizer = Adam(transformer.parameters(), lr=alpha, weight_decay=weight_decay)
        
        # Learning rate scheduler
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
        
        # Training loop with early stopping
        best_val_loss = float('inf')
        best_val_acc = 0
        patience = 5
        patience_counter = 0
        
        # For tracking metrics
        train_losses = []
        train_accs = []
        val_losses = []
        val_accs = []
        benign_accs = []
        malignant_accs = []
        
        print("Starting training...")
        for epoch in range(epochs):
            # Train for one epoch
            train_loss, train_acc = train_epoch(transformer, train_loader, optimizer, criterion, device)
            
            # Validate
            val_loss, val_acc, benign_acc, malignant_acc = validate(transformer, val_loader, criterion, device)
            
            # Update learning rate based on validation loss
            scheduler.step(val_loss)
            
            # Track metrics
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            benign_accs.append(benign_acc)
            malignant_accs.append(malignant_acc)
            
            # Print progress
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            print(f'  Benign Acc: {benign_acc:.2f}%, Malignant Acc: {malignant_acc:.2f}%')
            
            # Check if this is the best model so far - consider both classes
            # We use the average of benign and malignant accuracy to avoid bias towards majority class
            current_balanced_acc = (benign_acc + malignant_acc) / 2
            if current_balanced_acc > best_val_acc:
                best_val_acc = current_balanced_acc
                best_val_loss = val_loss
                patience_counter = 0
                
                # Save the best model
                torch.save(transformer.state_dict(), 'best_model.pth')
                print(f'  New best model saved with Balanced Acc: {best_val_acc:.2f}%')
            else:
                patience_counter += 1
                print(f'  Balanced accuracy did not improve. Patience: {patience_counter}/{patience}')
                
            # Early stopping
            if patience_counter >= patience:
                print(f'Early stopping triggered after {epoch+1} epochs')
                break
        
        # Load the best model for testing
        transformer.load_state_dict(torch.load('best_model.pth'))
        
        # Final evaluation on test set
        # transformer.eval()
        # test_result = validate(transformer, test_loader, criterion, device)
        # test_loss, test_acc, benign_acc, malignant_acc, report, conf_matrix, auc = test_result
        
        # print("\nFinal Test Results:")
        # print(f'Overall Accuracy: {test_acc:.2f}%')
        # print(f'Benign Accuracy: {benign_acc:.2f}%')
        # print(f'Malignant Accuracy: {malignant_acc:.2f}%')
        # print(f'Balanced Accuracy: {(benign_acc + malignant_acc) / 2:.2f}%')


        transformer.eval()
        test_loss, test_acc, benign_acc, malignant_acc = validate(transformer, test_loader, criterion, device)

        print("\nFinal Test Results:")
        print(f'Overall Accuracy: {test_acc:.2f}%')
        print(f'Benign Accuracy: {benign_acc:.2f}%')
        print(f'Malignant Accuracy: {malignant_acc:.2f}%')
        print(f'Balanced Accuracy: {(benign_acc + malignant_acc) / 2:.2f}%')





        # if auc is not None:
        #     print(f'AUC-ROC: {auc:.4f}')
        
        # print("\nConfusion Matrix:")
        # print(conf_matrix)
        
        # print("\nClassification Report:")
        # print(report)
    
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()