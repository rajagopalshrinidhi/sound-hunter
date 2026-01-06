#!/bin/bash
echo "Building Tesseract components..."
echo "Script starting in: $(pwd)"  # Add this line for debugging


cd ./components/audio_filter
tesseract build .
cd ../..

cd ./components/feature_extractor
tesseract build .
cd ../..

cd ./components/pattern_detector
tesseract build .
cd ../..

echo "Build complete! Run 'docker images' to see your components."