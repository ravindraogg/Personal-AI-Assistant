import torch
import torch.nn as nn
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path
from tqdm import tqdm  # For progress tracking

class TTSModel(nn.Module):
    def __init__(self, input_dim: int = 5, hidden_dim: int = 1024):
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

class TTSInference:
    def __init__(self, model_path: str, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        
        print(f"Using device: {device}")
        print("Loading model checkpoint...")
        self.checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        
        # Initialize model with correct dimensions
        self.model = TTSModel(input_dim=5, hidden_dim=1024)
        
        print("Loading model weights...")
        self.model.load_state_dict(self.checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()
        
        print("Model initialized successfully!")
    
    def preprocess_features(self, features: np.ndarray) -> np.ndarray:
        """Preprocess input features to match model requirements"""
        original_shape = features.shape
        
        # Handle different input shapes
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        elif len(features.shape) == 2:
            pass
        elif len(features.shape) == 3:
            # If we have a 3D array (e.g., mel spectrogram with channels)
            features = features.reshape(features.shape[0], -1)
        else:
            raise ValueError(f"Unsupported feature shape: {original_shape}")
        
        # If features dimension doesn't match, we need to adjust
        if features.shape[1] != 5:
            print(f"Reshaping features from {features.shape} to match input dimension (5)")
            
            if features.shape[1] > 5:
                features = features[:, :5]
            else:
                pad_width = ((0, 0), (0, 5 - features.shape[1]))
                features = np.pad(features, pad_width, mode='constant')
        
        return features
    
    def generate_speech(self, input_features: np.ndarray) -> np.ndarray:
        """Generate speech from input features"""
        try:
            # Preprocess features
            input_features = self.preprocess_features(input_features)
            print(f"Preprocessed features shape: {input_features.shape}")
            
            # Convert to tensor
            features_tensor = torch.FloatTensor(input_features).to(self.device)
            
            # Generate output
            print("Generating output...")
            with torch.no_grad():
                output = self.model(features_tensor)
                output = output.cpu().numpy()
            
            print(f"Generated output shape: {output.shape}")
            
            # Convert to audio
            print("Converting to audio...")
            audio = librosa.feature.inverse.mel_to_audio(
                output.T,
                sr=22050,
                n_iter=32
            )
            
            return audio
            
        except Exception as e:
            print(f"Error in generate_speech: {str(e)}")
            raise

def test_model():
    # Paths
    MODEL_PATH = r"D:\Data_Sets\Kaggle\Jarvis_voice_pack\extracted_features\tts_model.pth"
    OUTPUT_DIR = Path(r"D:\Data_Sets\Kaggle\Jarvis_voice_pack\generated_audio")
    FEATURE_DIR = Path(r"D:\Data_Sets\Kaggle\Jarvis_voice_pack\extracted_features")
    
    # Create output directory if it doesn't exist
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Initialize TTS system
    print("Initializing TTS system...")
    try:
        tts = TTSInference(MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return
    
    # Get all .npy files
    feature_files = list(FEATURE_DIR.glob('*.npy'))
    if not feature_files:
        print("No .npy files found in the directory.")
        return
    
    print(f"Found {len(feature_files)} feature files to process")
    
    # Process each file
    successful = 0
    failed = 0
    
    for feature_file in tqdm(feature_files, desc="Processing files"):
        print(f"\nProcessing: {feature_file.name}")
        
        try:
            # Load features
            input_features = np.load(feature_file)
            print(f"Loaded features with shape: {input_features.shape}")
            
            # Generate speech
            audio = tts.generate_speech(input_features)
            
            # Save audio
            output_path = OUTPUT_DIR / f"{feature_file.stem}_generated_speech.wav"
            sf.write(output_path, audio, 22050)
            print(f"✓ Successfully saved to {output_path}")
            
            successful += 1
            
        except Exception as e:
            print(f"✗ Error processing {feature_file.name}: {str(e)}")
            import traceback
            print(traceback.format_exc())
            failed += 1
    
    # Print summary
    print("\nProcessing Complete!")
    print(f"Successfully processed: {successful} files")
    print(f"Failed to process: {failed} files")
    print(f"Total files: {len(feature_files)}")

if __name__ == "__main__":
    print("Running batch processing of TTS features...")
    test_model()