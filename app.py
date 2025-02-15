import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple
import os
from pathlib import Path

class TTSDataset(Dataset):
    def __init__(self, feature_paths: List[str]):
        """
        Initialize the TTS dataset with only feature files
        Args:
            feature_paths: List of paths to .npy files containing audio features
        """
        self.feature_paths = feature_paths
        self._analyze_features()
    
    def _analyze_features(self):
        """Analyze features to determine dimensions and standardize format"""
        print("Analyzing features...")
        
        # Analyze first 100 files (or all if less than 100)
        sample_size = min(100, len(self.feature_paths))
        feature_dims = []
        sequence_lengths = []
        
        for path in self.feature_paths[:sample_size]:
            feature = np.load(path)
            if len(feature.shape) == 1:
                feature = feature.reshape(1, -1)
            sequence_lengths.append(feature.shape[0])
            feature_dims.append(feature.shape[1])
        
        # Set standard dimensions
        self.max_len = max(sequence_lengths)
        self.min_feature_dim = min(feature_dims)
        self.max_feature_dim = max(feature_dims)
        
        print(f"Sequence length range: {min(sequence_lengths)} to {max(sequence_lengths)}")
        print(f"Feature dimension range: {self.min_feature_dim} to {self.max_feature_dim}")
        print(f"Standardizing to sequence length: {self.max_len}")
        print(f"Standardizing to feature dimension: {self.min_feature_dim}")
    
    def standardize_feature(self, feature: np.ndarray) -> np.ndarray:
        """Standardize feature dimensions"""
        # Ensure 2D
        if len(feature.shape) == 1:
            feature = feature.reshape(1, -1)
        
        # Truncate or zero-pad sequence length
        if feature.shape[0] > self.max_len:
            feature = feature[:self.max_len, :]
        elif feature.shape[0] < self.max_len:
            padding = np.zeros((self.max_len - feature.shape[0], feature.shape[1]))
            feature = np.vstack([feature, padding])
        
        # Truncate feature dimension if necessary
        if feature.shape[1] > self.min_feature_dim:
            feature = feature[:, :self.min_feature_dim]
        
        return feature
    
    def __len__(self):
        return len(self.feature_paths)
    
    def __getitem__(self, idx) -> torch.Tensor:
        # Load and standardize features
        features = np.load(self.feature_paths[idx])
        features_standardized = self.standardize_feature(features)
        
        # Convert to tensor
        return torch.FloatTensor(features_standardized)

class TTSModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def train_model(model: nn.Module, train_loader: DataLoader, 
                num_epochs: int, device: str = 'cuda'):
    """Train the TTS model"""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, features in enumerate(train_loader):
            features = features.to(device)
            
            optimizer.zero_grad()
            output = model(features)
            
            loss = criterion(output, features)
            loss.backward()
            
            # Gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch} Average Loss: {avg_loss:.4f}')

def get_feature_paths(feature_dir: str) -> List[str]:
    """Get all .npy file paths from the directory"""
    feature_dir = Path(feature_dir)
    feature_paths = [str(path) for path in feature_dir.glob('*.npy')]
    return feature_paths

def main():
    # Your feature directory
    FEATURE_DIR = r"D:\Data_Sets\Kaggle\Jarvis_voice_pack\extracted_features"
    
    # Hyperparameters
    HIDDEN_DIM = 1024
    BATCH_SIZE = 8  # Reduced batch size
    NUM_EPOCHS = 50
    
    # Get dataset paths
    print("Loading dataset paths...")
    feature_paths = get_feature_paths(FEATURE_DIR)
    
    if not feature_paths:
        raise ValueError(f"No .npy files found in {FEATURE_DIR}")
    
    print(f"Found {len(feature_paths)} feature files")
    
    try:
        # Create dataset
        dataset = TTSDataset(feature_paths)
        
        # Create dataloader
        train_loader = DataLoader(
            dataset, 
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0,  # No multiprocessing for debugging
        )
        
        # Initialize model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        model = TTSModel(dataset.min_feature_dim, HIDDEN_DIM).to(device)
        
        # Train model
        train_model(model, train_loader, NUM_EPOCHS, device)
        
        # Save model
        save_path = os.path.join(FEATURE_DIR, 'tts_model.pth')
        torch.save({
            'model_state_dict': model.state_dict(),
            'feature_dim': dataset.min_feature_dim,
            'max_len': dataset.max_len
        }, save_path)
        print(f"Model saved to {save_path}")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()