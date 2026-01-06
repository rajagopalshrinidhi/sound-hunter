#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tesseract_core
import numpy as np
import json

def generate_sound(sound_type, duration=1, sample_rate=22050):
    """Generate different types of sounds for training"""
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    if sound_type == "bird":
        # High frequency chirp (2000-5000 Hz)
        freq = 3000 + 1000 * np.sin(10 * np.pi * t)  # Frequency modulation
        signal = np.sin(2 * np.pi * freq * t)
        
        # Add chirp pattern - ensure envelope matches signal length
        signal_len = len(signal)
        quarter = signal_len // 4
        remainder = signal_len % 4
        
        # Create envelope with exact length
        envelope = np.concatenate([
            np.ones(quarter),
            np.zeros(quarter),
            np.ones(quarter),
            np.zeros(quarter + remainder)  # Add remainder to last segment
        ])
        
        signal *= envelope
        
    elif sound_type == "motorcycle":
        # Low frequency rumble (80-200 Hz)
        signal = np.sin(2 * np.pi * 120 * t)  # Main engine frequency
        signal += 0.5 * np.sin(2 * np.pi * 240 * t)  # First harmonic
        signal += 0.3 * np.sin(2 * np.pi * 360 * t)  # Second harmonic
        
    elif sound_type == "whistle":
        # Pure tone (1000-2000 Hz)
        signal = np.sin(2 * np.pi * 1500 * t)
        
    else:  # Background noise
        signal = 0.5 * np.sin(2 * np.pi * 50 * t)  # Low hum
        signal += 0.3 * np.random.randn(len(t))    # White noise
    
    return signal

def train_detector(audio_samples, labels, target_sound, epochs=20):
    """Train the system to detect a specific sound"""
    print("\nTraining to detect: '%s'" % target_sound)
    print("Training samples: %d" % len(audio_samples))
    print("Positive examples: %d" % labels.count(target_sound))
    
    # Initialize filter parameters
    filter_params = {
        "low_freq": 500,   # Start in middle range
        "high_freq": 2000
    }
    
    # Collect target patterns for later detection
    target_patterns = []
    
    # Training parameters
    learning_rate = 50.0
    best_energy_ratio = 0
    best_params = filter_params.copy()
    
    # Load components as context managers
    with tesseract_core.Tesseract.from_image("audio-filter:latest") as audio_filter, \
         tesseract_core.Tesseract.from_image("feature-extractor:latest") as feature_extractor:
        
        for epoch in range(epochs):
            total_target_energy = 0
            total_other_energy = 0
            target_count = 0
            other_count = 0
            
            for audio, label in zip(audio_samples, labels):
                # Apply current filter
                filter_result = audio_filter.apply({
                    "audio_data": audio.tolist(),
                    "sample_rate": 22050,
                    "filter_params": filter_params
                })
                
                energy = filter_result["filter_energy"]
                
                # Track energies
                if label == target_sound:
                    total_target_energy += energy
                    target_count += 1
                    
                    # Extract features for target sounds
                    if epoch == epochs - 1:  # Last epoch
                        features = feature_extractor.apply({
                            "filtered_audio": filter_result["filtered_audio"],
                            "sample_rate": filter_result["sample_rate"]
                        })
                        target_patterns.append(features["feature_vector"])
                else:
                    total_other_energy += energy
                    other_count += 1
                
                # Compute loss
                if label == target_sound:
                    loss = -energy  # Want to maximize
                else:
                    loss = energy   # Want to minimize
                
                # Simple gradient-free optimization
                # (since jacobian has format issues)
                if epoch < epochs - 1:  # Don't update on last epoch
                    # Random walk optimization
                    if np.random.random() < 0.3:  # 30% chance to explore
                        # Try random perturbation
                        delta_low = np.random.uniform(-100, 100)
                        delta_high = np.random.uniform(-100, 100)
                        
                        # Apply perturbation
                        new_low = filter_params["low_freq"] + delta_low
                        new_high = filter_params["high_freq"] + delta_high
                        
                        # Ensure valid ranges
                        new_low = max(20, min(8000, new_low))
                        new_high = max(new_low + 50, min(10000, new_high))
                        
                        # Test new parameters
                        test_result = audio_filter.apply({
                            "audio_data": audio.tolist(),
                            "sample_rate": 22050,
                            "filter_params": {"low_freq": new_low, "high_freq": new_high}
                        })
                        
                        # Accept if better
                        new_energy = test_result["filter_energy"]
                        if (label == target_sound and new_energy > energy) or \
                           (label != target_sound and new_energy < energy):
                            filter_params["low_freq"] = new_low
                            filter_params["high_freq"] = new_high
            
            # Calculate energy ratio (want high ratio)
            avg_target = total_target_energy / max(target_count, 1)
            avg_other = total_other_energy / max(other_count, 1)
            energy_ratio = avg_target / max(avg_other, 1e-10)
            
            # Track best parameters
            if energy_ratio > best_energy_ratio:
                best_energy_ratio = energy_ratio
                best_params = filter_params.copy()
            
            if epoch % 5 == 0:
                print("Epoch %d: Filter: %.0f-%.0f Hz, Energy ratio: %.2f" % 
                      (epoch, filter_params['low_freq'], filter_params['high_freq'], energy_ratio))
    
    # Use best parameters
    filter_params = best_params
    print("\nTraining complete!")
    print("Optimal filter range: %.0f-%.0f Hz" % (filter_params['low_freq'], filter_params['high_freq']))
    print("Best energy ratio: %.2f" % best_energy_ratio)
    
    # Calculate average target pattern
    if target_patterns:
        avg_pattern = np.mean(target_patterns, axis=0).tolist()
    else:
        avg_pattern = [0.0] * 10
    
    return {
        "filter_params": filter_params,
        "target_pattern": avg_pattern,
        "sound_type": target_sound
    }

def main():
    print("=== Sound Hunter Training System ===")
    
    # Generate training data
    print("\nGenerating training data...")
    
    # Create diverse training set
    bird_sounds = [generate_sound("bird") for _ in range(5)]
    motorcycle_sounds = [generate_sound("motorcycle") for _ in range(5)]
    whistle_sounds = [generate_sound("whistle") for _ in range(5)]
    noise_sounds = [generate_sound("noise") for _ in range(5)]
    
    # Train for different sounds
    print("\n" + "="*50)
    print("TRAINING FOR BIRD DETECTION")
    print("="*50)
    
    all_sounds = bird_sounds + motorcycle_sounds + whistle_sounds + noise_sounds
    labels = (["bird"] * 5 + ["motorcycle"] * 5 + 
              ["whistle"] * 5 + ["noise"] * 5)
    
    bird_model = train_detector(all_sounds, labels, "bird")
    
    print("\n" + "="*50)
    print("TRAINING FOR MOTORCYCLE DETECTION")
    print("="*50)
    
    motorcycle_model = train_detector(all_sounds, labels, "motorcycle")
    
    print("\n" + "="*50)
    print("TRAINING FOR WHISTLE DETECTION")
    print("="*50)
    
    whistle_model = train_detector(all_sounds, labels, "whistle")
    
    # Save models
    print("\n" + "="*50)
    print("SAVING TRAINED MODELS")
    print("="*50)
    
    models = {
        "bird": bird_model,
        "motorcycle": motorcycle_model,
        "whistle": whistle_model
    }
    
    with open("trained_models.json", "w") as f:
        json.dump(models, f, indent=2)
    
    print("\n  Models saved to trained_models.json")
    print("\nSummary:")
    print("Bird detector:       %.0f-%.0f Hz" % (bird_model['filter_params']['low_freq'], bird_model['filter_params']['high_freq']))
    print("Motorcycle detector: %.0f-%.0f Hz" % (motorcycle_model['filter_params']['low_freq'], motorcycle_model['filter_params']['high_freq']))
    print("Whistle detector:    %.0f-%.0f Hz" % (whistle_model['filter_params']['low_freq'], whistle_model['filter_params']['high_freq']))

if __name__ == "__main__":
    main()