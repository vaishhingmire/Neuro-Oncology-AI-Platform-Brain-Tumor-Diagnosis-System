import numpy as np

def calibrate_confidence(logits: np.ndarray, temperature: float = 1.5) -> np.ndarray:
    """
    Implement temperature scaling for classifier logits to output calibrated probabilities.
    
    Args:
        logits (np.ndarray): Unscaled logits output by the classifier model.
        temperature (float): Scaling factor (T). T > 1 softens probabilities, T < 1 sharpens.
        
    Returns:
        np.ndarray: Calibrated probabilities.
    """
    # Ensure logits is a numpy array
    logits = np.array(logits, dtype=np.float32)
    
    # Apply temperature scaling
    scaled_logits = logits / temperature
    
    # Softmax to get probabilities
    exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=-1, keepdims=True)) # stability
    probabilities = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    
    return probabilities
