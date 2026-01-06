# Sound Hunter

A differentiable audio processing pipeline built with Tesseract for intelligent sound analysis and synthesis.

## Team

Shrinidhi Rajagopal

## License

Licensed under Apache 2.0


## Overview

Sound Hunter leverages Tesseract's differentiable programming capabilities to create an end-to-end pipeline for audio signal processing, enabling gradient-based optimization of sound detection, classification, and generation tasks.

## Features

- **Differentiable Audio Pipeline**: End-to-end gradient flow through audio processing components
- **Intelligent Sound Detection**: ML-powered audio event detection and classification  
- **Optimization-Driven Design**: Gradient-based parameter tuning for optimal audio processing
- **Real-time Processing**: Efficient pipeline execution for live audio streams

## Architecture

```
Audio Input → Preprocessing → Feature Extraction → ML Model → Post-processing → Output
     ↑                                                                           ↓
     └─────────────── Gradient Flow (Tesseract) ──────────────────────────────────┘
```

## Quick Start

### Prerequisites
- Python 3.10+
- Docker
- Tesseract framework
- Audio processing dependencies

### Installation

```bash
# Clone the repository
git clone https://github.com/rajagopalshrinidhi/sound-hunter.git
cd sound-hunter

# Run setup script (handles virtual environment and dependencies)
./run.sh
```

### Basic Usage

```markdown
```bash
# Run the Sound Hunter pipeline
./run.sh
```

The `run.sh` script performs the following tasks:
1. **Python Version Check**: Ensures Python 3.10+ is installed, and installs it if necessary.
2. **Virtual Environment Setup**: Creates and activates a virtual environment for dependency isolation.
3. **Dependency Installation**: Installs required Python packages from `requirements.txt`.
4. **Component Build**: Executes `build_components.sh` to build necessary components.
5. **Pipeline Testing**: Runs `test_pipeline.py` to verify basic functionality.
6. **Model Training**: Trains the system using `train_system.py`.
7. **Demo Retraining**: Demonstrates retraining with `demo_retrain.py`.
8. **Docker Cleanup**: Stops and removes conflicting Docker containers and images.

This script ensures a seamless setup and execution of the Sound Hunter pipeline.
```

## Tesseract Integration

This project demonstrates Tesseract's capabilities in:
- **Differentiable Signal Processing**: Maintaining gradients through audio transformations
- **Multi-scale Modeling**: Processing audio at different temporal resolutions
- **Inverse Problems**: Reconstructing audio sources from processed signals
- **Automated Design**: Optimizing filter parameters through gradient descent

## Technical Details

### Core Components
1. **Audio Preprocessor**: Differentiable audio normalization and windowing
2. **Feature Extractor**: Gradient-friendly spectral analysis
3. **Classification Network**: Deep learning model for sound identification
4. **Synthesis Module**: Differentiable audio generation

### Key Innovations
- Custom differentiable FFT implementation
- Gradient-preserving audio augmentation
- End-to-end optimization of the entire audio pipeline

## Results

[Add your results, metrics, and demonstrations here]

## Scientific Impact

Sound Hunter addresses real challenges in:
- Environmental monitoring through automated sound detection
- Assistive technologies for hearing-impaired individuals  
- Music information retrieval and audio content analysis
- Industrial noise monitoring and optimization

## Reproducibility

### Environment Setup
```bash
# Exact environment recreation
pip install -r requirements.txt
python -c "import tesseract; print(f'Tesseract version: {tesseract.__version__}')"
```

### Running Experiments
```bash
python test_pipeline.py    # Basic functionality tests
python train_system.py     # Model training
python demo_retrain.py     # Demo with retraining
```

## Future Work

- Integration with real-time audio streams
- Extension to multi-modal audio-visual processing
- Deployment as edge computing solution
- Enhanced differentiable audio synthesis
