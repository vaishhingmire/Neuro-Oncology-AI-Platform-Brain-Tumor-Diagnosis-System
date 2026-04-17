import torch
import torch.nn as nn
import numpy as np

class MorphologyMLP(nn.Module):
    def __init__(self, num_classes=4):
        super(MorphologyMLP, self).__init__()
        # Input features: area, eccentricity, compactness, boundary_irregularity, skull_proximity
        self.fc = nn.Sequential(
            nn.Linear(5, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes)
        )
        
    def forward(self, x):
        return self.fc(x)

def predict_morphology_probs(morph_model: nn.Module, morphology_features: dict) -> np.ndarray:
    """
    Get probability distribution from the morphology MLP.
    Expects morphology_features with keys:
    - area
    - eccentricity
    - compactness
    - boundary_irregularity
    - skull_proximity
    """
    if morph_model is None:
        return np.ones(4) / 4.0
        
    morph_model.eval()
    
    # Extract features in a specific order
    features = np.array([
        morphology_features.get("area", 0.0),
        morphology_features.get("eccentricity", 0.0),
        morphology_features.get("compactness", 0.0),
        morphology_features.get("boundary_irregularity", 0.0),
        morphology_features.get("skull_proximity", 0.0)
    ], dtype=np.float32)
    
    # Normalize features if needed (using arbitrary stable normalization for mock implementation)
    # Area can be large, others are mostly 0-1, compactness can be large
    features[0] = features[0] / 10000.0  # mock area normalization
    if features[2] > 100: features[2] = 100.0
    features[2] = features[2] / 100.0    # mock compactness normalization
    features[4] = features[4] / 256.0    # mock skull proximity normalization
    
    tensor_features = torch.tensor(features).unsqueeze(0)
    
    with torch.no_grad():
        logits = morph_model(tensor_features)
        probs = torch.softmax(logits, dim=1).squeeze(0).numpy()
        
    return probs

def fuse_predictions(cnn_probs: np.ndarray, morph_probs: np.ndarray, cnn_weight=0.7, morph_weight=0.3) -> np.ndarray:
    """
    Combines the CNN probabilities with the Morphology MLP probabilities.
    Final prediction = 0.7 * CNN probability + 0.3 * morphology probability
    """
    fused_probs = (cnn_probs * cnn_weight) + (morph_probs * morph_weight)
    
    # Ensure they sum to 1
    fused_probs = fused_probs / np.sum(fused_probs)
    return fused_probs
