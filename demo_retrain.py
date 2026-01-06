import tesseract_core
import numpy as np
import json

def test_detection(sound_type, model):
    """Test detection on a new sound"""
    # Generate test sound
    t = np.linspace(0, 1, 22050)
    
    if sound_type == "bird":
        test_audio = np.sin(2 * np.pi * 3500 * t) * np.random.choice([0, 1], size=len(t), p=[0.7, 0.3])
    elif sound_type == "motorcycle":
        test_audio = np.sin(2 * np.pi * 150 * t) + 0.5 * np.sin(2 * np.pi * 300 * t)
    elif sound_type == "whistle":
        test_audio = np.sin(2 * np.pi * 1600 * t)
    else:
        test_audio = 0.3 * np.random.randn(len(t))
    
    # Run through pipeline using context managers
    with tesseract_core.Tesseract.from_image("audio-filter:latest") as audio_filter:
        filter_result = audio_filter.apply({
            "audio_data": test_audio.tolist(),
            "sample_rate": 22050,
            "filter_params": model["filter_params"]
        })
    
    with tesseract_core.Tesseract.from_image("feature-extractor:latest") as feature_extractor:
        feature_result = feature_extractor.apply({
            "filtered_audio": filter_result["filtered_audio"],
            "sample_rate": filter_result["sample_rate"]
        })
    
    with tesseract_core.Tesseract.from_image("pattern-detector:latest") as pattern_detector:
        detection_result = pattern_detector.apply({
            "feature_vector": feature_result["feature_vector"],
            "target_pattern": model["target_pattern"],
            "detection_threshold": 0.7
        })
    
    return detection_result["is_match"], detection_result["confidence"]

def main():
    print("=== Demonstrating Retraining Capability ===\n")
    
    # Load trained models
    try:
        with open("trained_models.json", "r") as f:
            models = json.load(f)
    except:
        print("ERROR: Run train_system.py first to create models!")
        return
    
    # Test each detector
    test_sounds = ["bird", "motorcycle", "whistle", "noise"]
    
    print("Testing each detector on different sounds:")
    print("-" * 60)
    print("%-15s %-15s %-12s %-10s" % ("Detector", "Test Sound", "Detection", "Confidence"))
    print("-" * 60)
    
    for detector_name, model in models.items():
        print("\n%s DETECTOR (Filter: %.0f-%.0f Hz)" % 
              (detector_name.upper(), model['filter_params']['low_freq'], model['filter_params']['high_freq']))
        
        for test_sound in test_sounds:
            is_match, confidence = test_detection(test_sound, model)
            status = "  DETECTED" if is_match else "   Rejected"
            print("%-15s %-15s %-12s %.1f%%" % ("  ", test_sound, status, confidence * 100))
    
    print("\n" + "="*60)
    print("KEY INSIGHTS:")
    print("- Each detector learned different frequency ranges")
    print("- Same code, different training data = different behavior")
    print("- No hardcoded knowledge about sounds!")
    print("- System automatically discovered optimal parameters")

if __name__ == "__main__":
    main()