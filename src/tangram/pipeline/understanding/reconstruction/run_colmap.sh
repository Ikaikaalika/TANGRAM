#!/bin/bash

# COLMAP 3D Reconstruction Pipeline
# Usage: ./run_colmap.sh <input_directory> <output_directory>

INPUT_DIR=${1:-"../data/frames"}
OUTPUT_DIR=${2:-"../data/3d_points"}

echo "Running COLMAP reconstruction..."
echo "Input: $INPUT_DIR"
echo "Output: $OUTPUT_DIR"

# Check if COLMAP is installed
if ! command -v colmap &> /dev/null; then
    echo "Error: COLMAP not found. Install with: brew install colmap (macOS)"
    exit 1
fi

# Check if input directory exists and has images
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input directory $INPUT_DIR does not exist"
    exit 1
fi

IMAGE_COUNT=$(find "$INPUT_DIR" -name "*.jpg" -o -name "*.png" | wc -l)
if [ "$IMAGE_COUNT" -lt 2 ]; then
    echo "Error: Need at least 2 images for reconstruction. Found: $IMAGE_COUNT"
    echo "Run frame extraction first: python reconstruction/extract_frames.py"
    exit 1
fi

echo "Found $IMAGE_COUNT images for reconstruction"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Step 1: Feature extraction..."
colmap feature_extractor \
    --database_path "$OUTPUT_DIR/database.db" \
    --image_path "$INPUT_DIR" \
    --ImageReader.single_camera 1 \
    --ImageReader.camera_model PINHOLE \
    --SiftExtraction.max_image_size 1600

if [ $? -ne 0 ]; then
    echo "Error: Feature extraction failed"
    exit 1
fi

echo "Step 2: Feature matching..."
colmap exhaustive_matcher \
    --database_path "$OUTPUT_DIR/database.db" \
    --SiftMatching.guided_matching 1

if [ $? -ne 0 ]; then
    echo "Error: Feature matching failed"
    exit 1
fi

echo "Step 3: Structure from Motion..."
mkdir -p "$OUTPUT_DIR/sparse"
colmap mapper \
    --database_path "$OUTPUT_DIR/database.db" \
    --image_path "$INPUT_DIR" \
    --output_path "$OUTPUT_DIR/sparse" \
    --Mapper.ba_refine_focal_length 0 \
    --Mapper.ba_refine_principal_point 0

if [ $? -ne 0 ]; then
    echo "Error: Structure from Motion failed"
    exit 1
fi

# Check if sparse reconstruction was successful
if [ ! -d "$OUTPUT_DIR/sparse/0" ]; then
    echo "Error: Sparse reconstruction failed - no model generated"
    exit 1
fi

echo "Step 4: Dense reconstruction (optional)..."
mkdir -p "$OUTPUT_DIR/dense"

# Check available memory before dense reconstruction
AVAILABLE_MEM=$(free -m 2>/dev/null | awk 'NR==2{printf "%.1f", $7/1024}' || echo "unknown")
echo "Available memory: ${AVAILABLE_MEM}GB"

colmap image_undistorter \
    --image_path "$INPUT_DIR" \
    --input_path "$OUTPUT_DIR/sparse/0" \
    --output_path "$OUTPUT_DIR/dense" \
    --output_type COLMAP

if [ $? -eq 0 ]; then
    echo "Running patch match stereo..."
    colmap patch_match_stereo \
        --workspace_path "$OUTPUT_DIR/dense" \
        --PatchMatchStereo.max_image_size 1000

    if [ $? -eq 0 ]; then
        echo "Running stereo fusion..."
        colmap stereo_fusion \
            --workspace_path "$OUTPUT_DIR/dense" \
            --output_path "$OUTPUT_DIR/dense/fused.ply" \
            --StereoFusion.max_image_size 1000
    fi
fi

# Export sparse model to text format for easier parsing
echo "Exporting sparse model to text format..."
colmap model_converter \
    --input_path "$OUTPUT_DIR/sparse/0" \
    --output_path "$OUTPUT_DIR/sparse/0" \
    --output_type TXT

echo "COLMAP reconstruction complete!"
echo "Sparse model: $OUTPUT_DIR/sparse/0/"
echo "Dense point cloud: $OUTPUT_DIR/dense/fused.ply (if generated)"

# Show reconstruction statistics
if [ -f "$OUTPUT_DIR/sparse/0/cameras.txt" ]; then
    NUM_CAMERAS=$(grep -v '^#' "$OUTPUT_DIR/sparse/0/cameras.txt" | wc -l)
    echo "Cameras: $NUM_CAMERAS"
fi

if [ -f "$OUTPUT_DIR/sparse/0/images.txt" ]; then
    NUM_IMAGES=$(grep -v '^#' "$OUTPUT_DIR/sparse/0/images.txt" | grep -v '^$' | wc -l)
    NUM_IMAGES=$((NUM_IMAGES / 2))  # Each image has 2 lines
    echo "Registered images: $NUM_IMAGES"
fi

if [ -f "$OUTPUT_DIR/sparse/0/points3D.txt" ]; then
    NUM_POINTS=$(grep -v '^#' "$OUTPUT_DIR/sparse/0/points3D.txt" | wc -l)
    echo "3D points: $NUM_POINTS"
fi