#!/usr/bin/env bash
# Convert MedGemma HF model to GGUF Q4_K_M format
# Usage: ./scripts/convert_to_gguf.sh [HF_MODEL_DIR] [OUTPUT_DIR]

set -euo pipefail

HF_MODEL_DIR="${1:-data/models/medgemma-1.5-4b}"
OUTPUT_DIR="${2:-data/models}"
OUTPUT_FILE="$OUTPUT_DIR/medgemma-4b-it-Q4_K_M.gguf"
LLAMA_CPP_DIR="/tmp/llama.cpp"

echo "=== MedGemma GGUF Conversion ==="
echo "Source:  $HF_MODEL_DIR"
echo "Output:  $OUTPUT_FILE"
echo ""

# Check if output already exists
if [ -f "$OUTPUT_FILE" ]; then
    echo "GGUF file already exists at $OUTPUT_FILE"
    echo "Size: $(du -h "$OUTPUT_FILE" | cut -f1)"
    echo "Delete it first if you want to re-convert."
    exit 0
fi

# Check source model exists
if [ ! -d "$HF_MODEL_DIR" ]; then
    echo "ERROR: HF model directory not found: $HF_MODEL_DIR"
    echo ""
    echo "Download the model first:"
    echo "  cd backend && HF_TOKEN=<token> uv run python ../scripts/download_model.py"
    exit 1
fi

# Step 1: Check for pre-quantized GGUF on HuggingFace
echo "--- Step 1: Checking for pre-quantized GGUF ---"
echo "Checking HuggingFace for community-quantized GGUF models..."
echo "(MedGemma is relatively new - community GGUFs may not exist yet)"
echo ""

# Step 2: Clone llama.cpp if needed
echo "--- Step 2: Setting up llama.cpp ---"
if [ ! -d "$LLAMA_CPP_DIR" ]; then
    echo "Cloning llama.cpp..."
    git clone --depth 1 https://github.com/ggerganov/llama.cpp "$LLAMA_CPP_DIR"
else
    echo "llama.cpp already cloned at $LLAMA_CPP_DIR"
fi

# Install Python dependencies for conversion
echo "Installing conversion dependencies..."
pip install -q sentencepiece protobuf

# Step 3: Convert HF model to GGUF (f16)
echo ""
echo "--- Step 3: Converting HF → GGUF (f16) ---"
F16_FILE="$OUTPUT_DIR/google_medgemma-4b-it-f16.gguf"

python "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" \
    "$HF_MODEL_DIR" \
    --outfile "$F16_FILE" \
    --outtype f16

echo "F16 GGUF created: $(du -h "$F16_FILE" | cut -f1)"

# Step 4: Build llama.cpp quantize tool
echo ""
echo "--- Step 4: Building quantize tool ---"
cd "$LLAMA_CPP_DIR"
if [ ! -f "build/bin/llama-quantize" ]; then
    cmake -B build
    cmake --build build --target llama-quantize -j "$(nproc 2>/dev/null || sysctl -n hw.ncpu)"
fi
cd -

# Step 5: Quantize to Q4_K_M
echo ""
echo "--- Step 5: Quantizing f16 → Q4_K_M ---"
"$LLAMA_CPP_DIR/build/bin/llama-quantize" \
    "$F16_FILE" \
    "$OUTPUT_FILE" \
    Q4_K_M

echo ""
echo "=== Conversion Complete ==="
echo "Output: $OUTPUT_FILE"
echo "Size:   $(du -h "$OUTPUT_FILE" | cut -f1)"
echo ""

# Clean up f16 intermediate file
echo "Cleaning up intermediate f16 file..."
rm -f "$F16_FILE"

echo "Done! You can now run: docker compose up"
