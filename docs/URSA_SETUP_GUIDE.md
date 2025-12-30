# URSA HPC Setup Guide for STOFS-GNN Training

## Overview

URSA is a NOAA RDHPCS system located at NESCC in Fairmont, West Virginia. This guide covers setup for running STOFS-GNN surrogate model training on URSA's H100 GPUs.

## System Specifications

| Component | Specification |
|-----------|---------------|
| GPU Nodes | 58 nodes with 2x NVIDIA H100-NVL (94 GB each) |
| CPU | AMD Genoa 9654 (192 cores/node) |
| RAM | 384 GB per node (2 GB/core) |
| Interconnect | NDR-200 InfiniBand |
| OS | Rocky 9 |

## 1. Logging into URSA

### SSH Connection
```bash
ssh Mansur.Jisan@ursa-rsa.boulder.rdhpcs.noaa.gov
```

### Authentication
- Enter RSA PIN + Token code (no space between)
- Or use Yubikey PIN + OTP (long press)

### Login Message
Upon login, note your **Local Port** number (e.g., `56444`) - needed for file transfers.

## 2. File Systems and Directories

### Available File Systems

| Path | Type | Quota | Purge Policy | Notes |
|------|------|-------|--------------|-------|
| `/home/$USER` | Home | 10 GB | None | Limited space, not for data |
| `/scratch3`, `/scratch4` | Lustre | Project-based | None | Shared with Hera |
| `/scratch5/purged/$USER` | VAST | 250 TB | 30 days | Fast all-flash storage |

### Recommended Working Directory
```bash
# Create your working directory
mkdir -p /scratch5/purged/Mansur.Jisan/stofs_surrogate
cd /scratch5/purged/Mansur.Jisan/stofs_surrogate
mkdir -p data/processed_80k_option_a outputs/checkpoints scripts src
```

**Warning**: Files in `/scratch5/purged/` not accessed for 30 days are automatically deleted!

## 3. Checking Allocations and Partitions

### View Your Project Allocations
```bash
sacctmgr show associations user=$USER format=Account,Partition,QOS
```

Example output:
```
   Account  Partition                  QOS
---------- ---------- --------------------
   coastal            batch,debug,gpuwf,long
gpu-nos-surge         batch,debug,gpu,gpuwf
 nos-surge            batch,debug,gpuwf,long
```

### GPU Partitions

| Partition | GPU Type | GPUs/Node | QOS Access |
|-----------|----------|-----------|------------|
| u1-h100 | NVIDIA H100-NVL | 2 | `gpu` (priority), `gpuwf` (windfall) |
| u1-gh | NVIDIA GH200 | 1 | `gpuwf` only |
| u1-mi300x | AMD MI300X | 8 | `gpuwf` only |

### Check GPU Node Availability
```bash
sinfo -p u1-h100,u1-gh,u1-mi300x
```

### Check GPU Queue
```bash
squeue -p u1-h100
squeue -u $USER  # Your jobs only
```

## 4. Setting Up Python Environment

### Load Modules
```bash
module load python/3.11 cuda/12.8.1
```

### Create Virtual Environment
```bash
python -m venv ~/venv_stofs
source ~/venv_stofs/bin/activate
```

### Install PyTorch and Dependencies
```bash
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install torch-geometric torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.5.0+cu121.html
pip install numpy scipy matplotlib tqdm
```

### Verify Installation
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

Note: CUDA will show `False` on login nodes (no GPUs). It will work on compute nodes.

## 5. Transferring Data to URSA

### Method 1: Untrusted DTN (UDTN) with FileZilla

**FileZilla Settings:**
| Setting | Value |
|---------|-------|
| Host | `udtn-ursa.fairmont.rdhpcs.noaa.gov` |
| Port | 22 |
| Protocol | SFTP |
| User | `Mansur.Jisan` |
| Password | RSA PIN + Token |

**Remote Directory:** `/scratch3/Mansur.Jisan/` or `/scratch4/Mansur.Jisan/`

**After upload, copy from UDTN to working directory:**
```bash
# Copy NPZ files
cp /scratch3/data_untrusted/Mansur.Jisan/*.npz \
   /scratch5/purged/Mansur.Jisan/stofs_surrogate/data/processed_80k_option_a/

# Copy scripts
cp /scratch3/data_untrusted/Mansur.Jisan/*.py \
   /scratch5/purged/Mansur.Jisan/stofs_surrogate/scripts/

# Copy src folder
cp -r /scratch3/data_untrusted/Mansur.Jisan/src \
   /scratch5/purged/Mansur.Jisan/stofs_surrogate/
```

**Warning**: Files in `/scratch[3,4]/data_untrusted/` are purged after 5 days!

### Method 2: SSH Tunnel + FileZilla

1. Note your local port from login message (e.g., `56444`)
2. Keep SSH session open
3. Configure FileZilla:
   - Host: `localhost`
   - Port: Your local port (e.g., `56444`)
   - Protocol: SFTP

### Method 3: Globus (Recommended for Large Data)

1. Install Globus Connect Personal on your laptop
2. Go to https://app.globus.org
3. Login with "NOAA RDHPCS"
4. Transfer to endpoint: `noaardhpcs#ursa_untrusted`
5. Path: `/scratch3/Mansur.Jisan/` or `/scratch4/Mansur.Jisan/`

## 6. Submitting GPU Jobs

### SBATCH Script Template

```bash
#!/bin/bash
#SBATCH --job-name=stofs_80k
#SBATCH --account=gpu-nos-surge       # Your GPU project
#SBATCH --partition=u1-h100
#SBATCH --qos=gpu                     # Use 'gpuwf' for windfall
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:h100:1             # Request 1 H100 GPU
#SBATCH --time=24:00:00
#SBATCH --output=outputs/training_%j.log
#SBATCH --error=outputs/training_%j.err

cd /scratch5/purged/Mansur.Jisan/stofs_surrogate

# Load modules and activate environment
module load python/3.11 cuda/12.8.1
source ~/venv_stofs/bin/activate

# Environment variables
export STOFS_DATA_DIR=/scratch5/purged/Mansur.Jisan/stofs_surrogate/data/processed_80k_option_a
export STOFS_OUTPUT_DIR=/scratch5/purged/Mansur.Jisan/stofs_surrogate

echo "Starting training on $(hostname)"
nvidia-smi

python scripts/train_80k_inmemory.py
```

### Submit Job
```bash
sbatch run_ursa_h100.sh
```

### Job Commands
```bash
squeue -u $USER           # Check your jobs
scancel <job_id>          # Cancel a job
sacct -j <job_id>         # Job accounting info
scontrol show job <job_id> # Detailed job info
```

### Interactive GPU Session
```bash
salloc -A gpu-nos-surge -p u1-h100 -q gpu -N 1 --gres=gpu:h100:1 -t 60
```

## 7. Quick Reference Commands

```bash
# Check disk usage
df -h /scratch5/purged/Mansur.Jisan

# Check quota
quota -s

# List modules
module avail
module spider pytorch
module list

# GPU monitoring (on compute node)
nvidia-smi
watch -n 1 nvidia-smi

# Check job efficiency after completion
seff <job_id>
```

## 8. Useful Paths

| Description | Path |
|-------------|------|
| Home | `/home/Mansur.Jisan` |
| Working Directory | `/scratch5/purged/Mansur.Jisan/stofs_surrogate` |
| Training Data | `/scratch5/purged/Mansur.Jisan/stofs_surrogate/data/processed_80k_option_a` |
| Checkpoints | `/scratch5/purged/Mansur.Jisan/stofs_surrogate/outputs/checkpoints` |
| Python Environment | `~/venv_stofs` |
| UDTN Staging | `/scratch3/data_untrusted/Mansur.Jisan` |

## 9. Getting Help

- Submit help requests to: `rdhpcs.ursa.help@noaa.gov`
- URSA documentation: https://docs.rdhpcs.noaa.gov/systems/ursa_user_guide.html

## 10. Expected Training Performance

| Instance | GPU | RAM | Est. Time (150 epochs) |
|----------|-----|-----|------------------------|
| ParallelWorks g5.xlarge | A10G (24 GB) | 16 GB | ~195 hours (8 days) |
| **URSA H100** | H100 (94 GB) | 384 GB | **~10-20 hours** |

Using `train_80k_inmemory.py` on URSA loads all data into RAM for maximum training speed.
