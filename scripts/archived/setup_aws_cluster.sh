#!/bin/bash
# STOFS-GNN AWS Cluster Setup Script
# Run this on your ParallelWorks cluster after SSH login

set -e  # Exit on error

echo "=== STOFS-GNN Cluster Setup ==="
echo "Step 1: Installing NVIDIA drivers and CUDA..."

# Install NVIDIA drivers (Rocky 8)
sudo dnf install -y epel-release
sudo dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo
sudo dnf install -y cuda-drivers cuda-toolkit-12-2

# Set CUDA environment
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

echo "Step 2: Installing Miniconda..."
wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda3
rm miniconda.sh
$HOME/miniconda3/bin/conda init bash
source ~/.bashrc

echo "Step 3: Creating conda environment..."
conda create -n stofs python=3.10 -y
source activate stofs

echo "Step 4: Installing PyTorch with CUDA..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo "Step 5: Installing PyTorch Geometric..."
pip install torch-geometric
pip install pyg_lib torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-2.1.0+cu121.html

echo "Step 6: Cloning STOFS-GNN repository..."
git clone https://github.com/mansurjisan/stofs_surrogate.git
cd stofs_surrogate
pip install -r requirements.txt

echo "Step 7: Verifying installation..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
python -c "import torch_geometric; print(f'PyG: {torch_geometric.__version__}')"

echo ""
echo "=== Setup Complete! ==="
echo "Next steps:"
echo "1. Upload your preprocessed .npz data files to ~/stofs_surrogate/data/processed/"
echo "2. Run: conda activate stofs"
echo "3. Run: python scripts/train_cwl_gnn_optimized_v3.py --train"
