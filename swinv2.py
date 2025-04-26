import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import timm
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score

# 1. TopoRET
class TopoRET(nn.Module):
    def __init__(self, num_classes, img_feat_dim, embed_dim=96):
        super(TopoRET, self).__init__()
        self.embed_dim = embed_dim
        
        self.swin = timm.create_model(
            'swinv2_tiny_window8_256',  
            pretrained=True,
            num_classes=0,
            img_size=256   
        )

        if img_feat_dim != embed_dim:
            self.img_proj = nn.Linear(img_feat_dim, embed_dim)
        else:
            self.img_proj = nn.Identity()
        
        # Embed each PH feature into embed_dim
        self.PH_embed = nn.Linear(1, embed_dim)  # Each scalar

        # PH self-attention
        self.self_attn_PH = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=4, batch_first=True)
        
        # Fusion MLP
        self.fusion_mlp = nn.Sequential(
            nn.Linear(2 * embed_dim, 3 * embed_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(3 * embed_dim, 3 * embed_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        
        self.classifier = nn.Linear(3 * embed_dim, num_classes)

    def forward(self, image, PH):
        # Image features from SwinV2
        img_features = self.swin.forward_features(image) 
        B, C, H, W = img_features.shape
        img_tokens = img_features.flatten(2).transpose(1, 2) 
        img_tokens = self.img_proj(img_tokens) 
        
        # PH embedding: treat each of Betti Number as a separate token
        PH_reshaped = PH.unsqueeze(-1)  # Shape: (B, 100, 1)
        PH_embedded = self.PH_embed(PH_reshaped)  # Shape: (B, 100, embed_dim)

        # PH self-attention
        PH_self, _ = self.self_attn_PH(query=PH_embedded, key=PH_embedded, value=PH_embedded)
        
        # Pooling
        img_pooled = img_tokens.mean(dim=1)  # Shape: (B, embed_dim)
        PH_self_pooled = PH_self.mean(dim=1)  # Shape: (B, embed_dim)

        # Fusion MLP
        fused_features = torch.cat([img_pooled, PH_self_pooled], dim=-1)
        fused = self.fusion_mlp(fused_features)
        
        logits = self.classifier(fused)
        return logits

# Dataset
class ImageNPZPHDataset(Dataset):
    def __init__(self, npz_file, PH_file, transform=None):
        # Images
        npz_data = np.load(npz_file)
        self.images = npz_data['data']
        
        # Betti Vectors
        self.PH_data = pd.read_csv(PH_file)
        
        # Extract Betti vectors
        feature_cols = [str(i) for i in range(100)]
        self.PH_features = self.PH_data[feature_cols].values.astype(np.float32)
        
        # Extract labels
        self.labels = self.PH_data['label'].values.astype(np.int64)
        
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]  # NumPy array
        
        # Ensure image is 2D (H, W)
        if image.ndim == 3 and image.shape[0] == 1:
            image = image.squeeze(0)  # Convert (1, H, W) to (H, W)
        
        # Convert to 3 channels by repeating the grayscale channel
        image = np.stack([image] * 3, axis=0)  # Shape: (3, H, W)
        image = image.transpose(1, 2, 0)  # Convert to (H, W, 3) for ToTensor
        
        if self.transform:
            image = self.transform(image)
        
        PH_feat = torch.tensor(self.PH_features[idx], dtype=torch.float)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, PH_feat, label

# Splitting & Dataloaders
def create_dataloaders(npz_file, PH_file, batch_size=64, transform=None):
    full_dataset = ImageNPZPHDataset(npz_file, PH_file, transform=transform)
    
    train_dataset, test_dataset = train_test_split(full_dataset, test_size=0.2)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, test_loader

# Training
def train_model(model, train_loader, num_epochs, device):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 75], gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, PH_feats, labels in train_loader:
            images = images.to(device)
            PH_feats = PH_feats.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images, PH_feats)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")
        scheduler.step()

if __name__ == '__main__':
    # Path to files
    npz_file = 'cdr4.npz'
    PH_file = 'CDR4.csv'
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    
    train_loader, test_loader = create_dataloaders(
        npz_file, PH_file, batch_size=128, transform=transform
    )
    
    # Create a dummy image to match our input resolution
    dummy_image = torch.randn(1, 3, 256, 256)
    swin_dummy = timm.create_model(
        'swinv2_tiny_window8_256',
        pretrained=True,
        num_classes=0,
        img_size=256   
    )

    with torch.no_grad():
        features = swin_dummy.forward_features(dummy_image)
    _, C, _, _ = features.shape
    img_feat_dim = C  # Used to initialize projection layer
    
    num_classes = 4
    model = TopoRET(num_classes=num_classes, img_feat_dim=img_feat_dim, embed_dim=128)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_epochs = 100
    
    train_model(model, train_loader, num_epochs=num_epochs, device=device)
    
    # Test
    model.eval()
    all_labels = []
    all_probs = []
    all_preds = []
    
    with torch.no_grad():
        for images, PH_feats, labels in test_loader:
            images = images.to(device)
            PH_feats = PH_feats.to(device)
            labels = labels.to(device)
            outputs = model(images, PH_feats)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    
    if num_classes > 2:
        accuracy = accuracy_score(all_labels, all_preds)
        roc_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')
        precision = precision_score(all_labels, all_preds, average='weighted')
        recall = recall_score(all_labels, all_preds, average='weighted')
        
        print(f"\nTest Accuracy: {accuracy:.4f}")
        print(f"Test ROC-AUC: {roc_auc:.4f}")
        print(f"Test Precision: {precision:.4f}")
        print(f"Test Recall: {recall:.4f}")
    else:
        accuracy = accuracy_score(all_labels, all_preds)
        roc_auc = roc_auc_score(all_labels, all_probs)
        precision = precision_score(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds)
        
        print("Test Accuracy: {:.4f}".format(accuracy))
        print("Test ROC-AUC: {:.4f}".format(roc_auc))
        print("Test Precision: {:.4f}".format(precision))
        print("Test Recall: {:.4f}".format(recall))