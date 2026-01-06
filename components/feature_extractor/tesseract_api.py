from typing import Dict, Any
import numpy as np
from pydantic import BaseModel, Field

# Handle imports for build time
try:
    from tesseract_core.runtime import Array, Differentiable, Float32
except ImportError:
    from typing import TypeVar, Generic
    T = TypeVar('T')
    class Differentiable(Generic[T]):
        pass
    Float32 = float
    def Array(shape, dtype):
        return list

class InputSchema(BaseModel):
    filtered_audio: Differentiable[Array[(None,), Float32]] = Field(
        description="Filtered audio signal from previous stage"
    )
    sample_rate: int = Field(
        description="Sample rate in Hz",
        ge=8000
    )

class AudioFeatures(BaseModel):
    rms_energy: Differentiable[Float32] = Field(description="Root mean square energy")
    peak_amplitude: Differentiable[Float32] = Field(description="Maximum absolute amplitude")
    zero_crossing_rate: Float32 = Field(description="Zero crossing rate")
    spectral_centroid: Float32 = Field(description="Spectral centroid frequency")

class OutputSchema(BaseModel):
    features: AudioFeatures = Field(
        description="Extracted audio features"
    )
    feature_vector: Differentiable[Array[(10,), Float32]] = Field(
        description="Feature vector for pattern matching"
    )

def apply(inputs: InputSchema) -> Dict[str, Any]:
    """Extract acoustic features from filtered audio"""
    # Extract from Pydantic model
    audio = np.array(inputs.filtered_audio)
    sample_rate = inputs.sample_rate
    
    # Compute features
    if len(audio) > 0:
        rms_energy = float(np.sqrt(np.mean(audio**2)))
        peak_amplitude = float(np.max(np.abs(audio)))
    else:
        rms_energy = 0.0
        peak_amplitude = 0.0
    
    # Zero crossing rate
    if len(audio) > 1:
        zero_crossings = np.sum(np.diff(np.sign(audio)) != 0)
        zero_crossing_rate = float(zero_crossings / len(audio))
    else:
        zero_crossing_rate = 0.0
    
    # Spectral features
    if len(audio) > 0:
        fft = np.fft.rfft(audio)
        freqs = np.fft.rfftfreq(len(audio), 1/sample_rate)
        magnitude = np.abs(fft)
        
        if np.sum(magnitude) > 0:
            spectral_centroid = float(np.sum(freqs * magnitude) / np.sum(magnitude))
        else:
            spectral_centroid = 0.0
    else:
        spectral_centroid = 0.0
    
    # Create feature vector
    feature_vector = [
        rms_energy,
        peak_amplitude,
        zero_crossing_rate,
        spectral_centroid / 10000.0,  # Normalize
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0  # Padding to 10 features
    ]
    
    # Return as dictionary
    return {
        "features": {
            "rms_energy": rms_energy,
            "peak_amplitude": peak_amplitude,
            "zero_crossing_rate": zero_crossing_rate,
            "spectral_centroid": spectral_centroid
        },
        "feature_vector": feature_vector
    }

def schema_input():
    return InputSchema.model_json_schema()

def schema_output():
    return OutputSchema.model_json_schema()