#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tesseract_core
import numpy as np

def generate_test_signal(frequency, duration=2, sample_rate=22050):
    """Generate a test sine wave"""
    t = np.linspace(0, duration, int(sample_rate * duration))
    return np.sin(2 * np.pi * frequency * t)

def test_pipeline():
    print("Loading Tesseract components...")
    
    # Generate test audio (440 Hz - musical note A)
    sample_rate = 22050
    test_audio = generate_test_signal(440, duration=1, sample_rate=sample_rate)
    
    # Add some noise
    test_audio += 0.1 * np.random.randn(len(test_audio))
    
    print("\nRunning pipeline on 440 Hz test signal...")
    
    # Use components as context managers
    with tesseract_core.Tesseract.from_image("audio-filter:latest") as audio_filter:
        # Step 1: Filter
        filter_params = {
            "low_freq": 400,
            "high_freq": 500
        }
        
        filter_result = audio_filter.apply({
            "audio_data": test_audio.tolist(),
            "sample_rate": sample_rate,
            "filter_params": filter_params
        })
        
        print("✓ Filter applied")
        print("  - Energy: %.2f" % filter_result['filter_energy'])
        print("  - Peak frequency: %.2f Hz" % filter_result['peak_frequency'])
    
    with tesseract_core.Tesseract.from_image("feature-extractor:latest") as feature_extractor:
        # Step 2: Extract features
        feature_result = feature_extractor.apply({
            "filtered_audio": filter_result["filtered_audio"],
            "sample_rate": filter_result["sample_rate"]
        })
        
        print("✓ Features extracted: %d features" % len(feature_result['feature_vector']))
    
    with tesseract_core.Tesseract.from_image("pattern-detector:latest") as pattern_detector:
        # Step 3: Pattern detection (comparing to itself for demo)
        detection_result = pattern_detector.apply({
            "feature_vector": feature_result["feature_vector"],
            "target_pattern": feature_result["feature_vector"],
            "detection_threshold": 0.8
        })
        
        print("✓ Pattern detection complete")
        print("  - Match: %s" % detection_result['is_match'])
        print("  - Confidence: %.1f%%" % (detection_result['confidence'] * 100))
    
    # # Test gradient computation
    # print("\n--- Testing gradient computation ---")
    # try:
    #     with tesseract_core.Tesseract.from_image("audio-filter:latest") as audio_filter:
    #         # Try different jacobian input formats
    #         try:
    #             # First try: Use exact nested field names
    #             gradients = audio_filter.jacobian(
    #                 {
    #                     "audio_data": test_audio.tolist(),
    #                     "sample_rate": sample_rate,
    #                     "filter_params": filter_params
    #                 },
    #                 ["filter_params.low_freq", "filter_params.high_freq"],
    #                 ["filter_energy"]
    #             )
    #             print("✓ Gradients computed (nested fields):")
    #         except Exception as e1:
    #             # Second try: Just the parent field
    #             try:
    #                 gradients = audio_filter.jacobian(
    #                     {
    #                         "audio_data": test_audio.tolist(),
    #                         "sample_rate": sample_rate,
    #                         "filter_params": filter_params
    #                     },
    #                     ["filter_params"],
    #                     ["filter_energy"]
    #                 )
    #                 print("✓ Gradients computed (parent field):")
    #             except Exception as e2:
    #                 print(f"⚠ Gradient computation format issue")
    #                 print(f"  Error 1: {e1}")
    #                 print(f"  Error 2: {e2}")
    #                 print("\n✓ Main pipeline works! Gradient computation is optional.")
    #                 return True
            
    #         # Display gradients if successful
    #         if "jacobian" in gradients:
    #             for param, grads in gradients["jacobian"].items():
    #                 if isinstance(grads, dict):
    #                     for output, value in grads.items():
    #                         print(f"  - d({output})/d({param}) = {value}")
    #                 else:
    #                     print(f"  - {param}: {grads}")
    #         else:
    #             print("  - Result:", gradients)
                
    # except Exception as e:
    #     print(f"⚠ Gradient computation not available: {e}")
    #     print("\n✓ Main pipeline works! Gradient computation is optional.")
    
    return True

if __name__ == "__main__":
    test_pipeline()