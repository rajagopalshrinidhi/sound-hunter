#!/usr/bin/env python3

import tesseract_core
import numpy as np
import json

# Generate test data
sample_rate = 22050
t = np.linspace(0, 1, sample_rate)
test_audio = np.sin(2 * np.pi * 440 * t) + 0.1 * np.random.randn(len(t))

print("Testing jacobian call...")

with tesseract_core.Tesseract.from_image("audio-filter:latest") as audio_filter:
    # First, let's see what the apply method returns
    apply_result = audio_filter.apply({
        "audio_data": test_audio.tolist(),
        "sample_rate": sample_rate,
        "filter_params": {"low_freq": 400, "high_freq": 500}
    })
    print(f"\nApply result keys: {apply_result.keys()}")
    
    # Now try jacobian with different approaches
    print("\n--- Trying jacobian calls ---")
    
    # Approach 1: What we've been trying
    try:
        print("\n1. Standard call:")
        result = audio_filter.jacobian(
            {
                "audio_data": test_audio.tolist(),
                "sample_rate": sample_rate,
                "filter_params": {"low_freq": 400, "high_freq": 500}
            },
            ["filter_params.low_freq", "filter_params.high_freq"],
            ["filter_energy"]
        )
        print("Success! Result:", json.dumps(result, indent=2))
    except Exception as e:
        print(f"Failed: {type(e).__name__}: {str(e)[:200]}...")
    
    # Approach 2: Try with just the parent field
    try:
        print("\n2. Parent field only:")
        result = audio_filter.jacobian(
            {
                "audio_data": test_audio.tolist(),
                "sample_rate": sample_rate,
                "filter_params": {"low_freq": 400, "high_freq": 500}
            },
            ["filter_params"],
            ["filter_energy"]
        )
        print("Success! Result:", json.dumps(result, indent=2))
    except Exception as e:
        print(f"Failed: {type(e).__name__}: {str(e)[:200]}...")
    
    # Approach 3: Try different input fields
    try:
        print("\n3. Audio data gradient:")
        result = audio_filter.jacobian(
            {
                "audio_data": test_audio.tolist(),
                "sample_rate": sample_rate,
                "filter_params": {"low_freq": 400, "high_freq": 500}
            },
            ["audio_data"],
            ["filter_energy"]
        )
        print("Success! Result type:", type(result))
        if isinstance(result, dict) and "jacobian" in result:
            print("Jacobian keys:", list(result["jacobian"].keys()))
    except Exception as e:
        print(f"Failed: {type(e).__name__}: {str(e)[:200]}...")