#!/bin/bash
#SBATCH --job-name=nnssl_openneuro
#SBATCH --nodes=6
#SBATCH --ntasks-per-node=4              # 4 tasks = 4 GPUs per node
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=12                # or however many you need
#SBATCH --mem=256G                        # adjust as needed
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Load any required modules
export nnssl_raw=/scratch/e/ekacz/nnssl/nnssl_raw
export nnssl_preprocessed=/scratch/e/ekacz/nnssl/nnssl_preprocessed
export nnssl_results=/scratch/e/ekacz/nnssl/nnssl_results

# Activate your virtualenv or conda
source /home/e/ekacz/MICCAI_challenges/nnssl-openneuro-temp/temp-nnssl/bin/activate

# Optional: Set a fixed port if not using rendezvous
#MASTER_PORT=29601

# Get master address
#MASTER_ADDR=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n 1)

MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_PORT=$((12000 + RANDOM % 10000))


# Torchrun: 2 nodes * 4 gpus = 8 processes
# torchrun \
#   --nnodes=2 \
#   --nproc_per_node=4 \
#   --node_rank=$SLURM_NODEID \
#   --rdzv_backend=c10d \
#   --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
#   -m nnssl.run.run_training \
#   745 'onemmiso' -tr SimCLRTrainer -p nnsslPlans -num_gpus 8

srun --ntasks-per-node=1 --nodes=6 bash -c "
echo Node=\$(hostname), SLURM_NODEID=\$SLURM_NODEID
torchrun \
  --nproc_per_node=4 \
  --nnodes=6 \
  --node_rank=\$SLURM_NODEID \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  -m nnssl.run.run_training \
  745 'onemmiso' -tr SimCLRTrainer -p nnsslPlans -num_gpus 24
"
