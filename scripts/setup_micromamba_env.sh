#!/bin/bash
# ==============================================================================
# YOLO_ChestXray Micromamba Python 3.10 Isolation Builder (Plan 1_4 Standard)
# Installs an isolated, fully reproducible Python 3.10 + Torch 2.0.1 environment
# ==============================================================================

set -e

echo "🚀 [Plan 1_4] Initializing Micromamba Isolated Environment Builder..."

# 1. Install Micromamba binary if not present
if ! command -v micromamba &> /dev/null; then
    echo "📦 Downloading Micromamba portable binary..."
    curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C /usr/local/bin --strip-components=1 bin/micromamba
fi

export MAMBA_ROOT_PREFIX="/content/micromamba"
mkdir -p "$MAMBA_ROOT_PREFIX"

# 2. Create isolated Python 3.10 environment
ENV_NAME="yolov7_p310"
if [ ! -d "$MAMBA_ROOT_PREFIX/envs/$ENV_NAME" ]; then
    echo "🔨 Creating isolated environment '$ENV_NAME' (Python 3.10)..."
    micromamba create -y -n "$ENV_NAME" -c conda-forge python=3.10 pip numpy=1.23.5
fi

# 3. Install Torch 2.0.1 + CUDA 11.8 and exact compatibility dependencies
echo "📦 Installing Torch 2.0.1 + TorchVision 0.15.2 + cu118..."
micromamba run -n "$ENV_NAME" pip install --no-cache-dir \
    torch==2.0.1+cu118 \
    torchvision==0.15.2+cu118 \
    --extra-index-url https://download.pytorch.org/whl/cu118

# 4. Install repository requirements
echo "📦 Installing YOLO_ChestXray dependencies..."
micromamba run -n "$ENV_NAME" pip install --no-cache-dir -r requirements.txt

# 5. Apply compatibility patches
echo "🔧 Applying YOLOv7 PyTorch & NumPy compatibility patches..."
micromamba run -n "$ENV_NAME" python scripts/fix_yolov7_env.py ./

echo "=============================================================================="
echo "✅ [Plan 1_4] Isolated Python 3.10 + Torch 2.0.1 Environment Ready!"
echo "👉 Use: micromamba run -n $ENV_NAME python <script.py>"
echo "=============================================================================="
