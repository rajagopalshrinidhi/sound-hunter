from typing import Dict, Any, List
import numpy as np
import json
import traceback
from pydantic import BaseModel, Field, field_validator
from typing_extensions import Self

# Handle imports for build time
try:
    from tesseract_core.runtime import Array, Differentiable, Float32
except ImportError:
    # Fallback for build time
    from typing import TypeVar, Generic
    T = TypeVar('T')
    class Differentiable(Generic[T]):
        pass
    Float32 = float
    def Array(shape, dtype):
        return list

class FilterParameters(BaseModel):
    low_freq: Differentiable[Float32] = Field(
        description="Lower frequency cutoff in Hz",
        ge=20.0,
        le=10000.0
    )
    high_freq: Differentiable[Float32] = Field(
        description="Upper frequency cutoff in Hz", 
        ge=20.0,
        le=10000.0
    )
    
    @field_validator('high_freq')
    @classmethod
    def validate_freq_range(cls, v: float, info) -> float:
        if 'low_freq' in info.data and v <= info.data['low_freq']:
            raise ValueError('high_freq must be greater than low_freq')
        return v

class InputSchema(BaseModel):
    audio_data: Differentiable[Array[(None,), Float32]] = Field(
        description="Audio signal samples"
    )
    sample_rate: int = Field(
        description="Sample rate in Hz",
        ge=8000
    )
    filter_params: FilterParameters = Field(
        description="Bandpass filter parameters"
    )

class OutputSchema(BaseModel):
    filtered_audio: Differentiable[Array[(None,), Float32]] = Field(
        description="Filtered audio signal"
    )
    filter_energy: Differentiable[Float32] = Field(
        description="Total energy of filtered signal"
    )
    peak_frequency: Float32 = Field(
        description="Dominant frequency"
    )
    sample_rate: int = Field(
        description="Sample rate"
    )

def log_debug(msg):
    """Log to both console and file for debugging"""
    print(f"[JACOBIAN DEBUG] {msg}")
    try:
        with open("/tmp/jacobian_debug.log", "a") as f:
            f.write(f"{msg}\n")
    except:
        pass

def apply(inputs: InputSchema) -> Dict[str, Any]:
    """Apply bandpass filter to audio signal"""
    # Extract data from Pydantic model
    audio_data = np.array(inputs.audio_data)
    sample_rate = inputs.sample_rate
    low_freq = float(inputs.filter_params.low_freq)
    high_freq = float(inputs.filter_params.high_freq)
    
    # Simple frequency domain filtering
    fft = np.fft.rfft(audio_data)
    freqs = np.fft.rfftfreq(len(audio_data), 1/sample_rate)
    
    # Find peak frequency
    if len(fft) > 0:
        peak_idx = np.argmax(np.abs(fft))
        peak_freq = float(freqs[peak_idx])
    else:
        peak_freq = 0.0
    
    # Apply filter in frequency domain
    mask = (freqs >= low_freq) & (freqs <= high_freq)
    fft_filtered = fft * mask
    
    # Convert back to time domain
    filtered_audio = np.fft.irfft(fft_filtered, len(audio_data))
    
    # Compute energy
    filter_energy = float(np.sum(filtered_audio ** 2))
    
    # Return as dictionary
    return {
        "filtered_audio": filtered_audio.tolist(),
        "filter_energy": filter_energy,
        "peak_frequency": peak_freq,
        "sample_rate": sample_rate
    }

def jacobian(inputs: InputSchema, jac_inputs: List[str], jac_outputs: List[str]) -> Dict[str, Any]:
    """Compute gradients for filter optimization"""
    try:
        log_debug("="*50)
        log_debug(f"jacobian called!")
        log_debug(f"inputs type: {type(inputs)}")
        log_debug(f"jac_inputs: {jac_inputs}")
        log_debug(f"jac_outputs: {jac_outputs}")
        
        # Try to understand the input structure
        if hasattr(inputs, 'filter_params'):
            log_debug(f"inputs.filter_params.low_freq: {inputs.filter_params.low_freq}")
            log_debug(f"inputs.filter_params.high_freq: {inputs.filter_params.high_freq}")
        
        # Initialize result
        result = {}
        
        # Simple test: return minimal structure
        if "filter_energy" in jac_outputs:
            for jac_input in jac_inputs:
                log_debug(f"Processing jac_input: {jac_input}")
                
                if jac_input == "filter_params.low_freq":
                    result[jac_input] = {"filter_energy": -0.1}
                elif jac_input == "filter_params.high_freq":
                    result[jac_input] = {"filter_energy": 0.1}
                elif jac_input == "audio_data":
                    # For array inputs, gradient should be an array
                    audio_len = len(inputs.audio_data)
                    # Return gradient as array
                    gradient_array = [0.0] * audio_len
                    gradient_array[0] = 0.001  # Just a test value
                    result[jac_input] = {"filter_energy": gradient_array}
        
        log_debug(f"Returning result keys: {list(result.keys())}")
        log_debug(f"Result structure: {json.dumps(result, default=str)[:500]}")
        
        return result
        
    except Exception as e:
        log_debug(f"ERROR in jacobian: {type(e).__name__}: {str(e)}")
        log_debug(f"Traceback: {traceback.format_exc()}")
        raise

def schema_input():
    return InputSchema.model_json_schema()

def schema_output():
    return OutputSchema.model_json_schema()