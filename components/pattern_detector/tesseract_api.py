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
    feature_vector: Differentiable[Array[(10,), Float32]] = Field(
        description="Feature vector from audio analysis"
    )
    target_pattern: Array[(10,), Float32] = Field(
        description="Target pattern to match against"
    )
    detection_threshold: Float32 = Field(
        description="Similarity threshold for detection",
        default=0.8,
        ge=0.0,
        le=1.0
    )

class OutputSchema(BaseModel):
    is_match: bool = Field(
        description="Whether the pattern matches above threshold"
    )
    similarity_score: Differentiable[Float32] = Field(
        description="Cosine similarity between vectors"
    )
    confidence: Differentiable[Float32] = Field(
        description="Detection confidence (0-1)"
    )

def apply(inputs: InputSchema) -> Dict[str, Any]:
    """Detect if audio matches target pattern"""
    # Extract from Pydantic model
    feature_vector = np.array(inputs.feature_vector)
    target_pattern = np.array(inputs.target_pattern)
    threshold = float(inputs.detection_threshold)
    
    # Normalize vectors
    feature_norm = np.linalg.norm(feature_vector)
    target_norm = np.linalg.norm(target_pattern)
    
    if feature_norm > 0 and target_norm > 0:
        # Compute cosine similarity
        similarity = float(
            np.dot(feature_vector, target_pattern) / (feature_norm * target_norm)
        )
    else:
        similarity = 0.0
    
    # Detection decision
    is_match = bool(similarity > threshold)
    
    # Confidence score
    confidence = float(max(0.0, min(1.0, similarity)))
    
    # Return as dictionary
    return {
        "is_match": is_match,
        "similarity_score": similarity,
        "confidence": confidence
    }

def schema_input():
    return InputSchema.model_json_schema()

def schema_output():
    return OutputSchema.model_json_schema()