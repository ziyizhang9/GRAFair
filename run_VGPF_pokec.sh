#!/bin/bash
#SBATCH --time=3:00:00
#SBATCH --account=rrg-baochun
#SBATCH --output=run_VGPF_pokec.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=32000M
module load StdEnv/2020  
module load gcc/8.4.0
module load cuda/10.2
export LD_LIBRARY_PATH=/cvmfs/soft.computecanada.ca/easybuild/software/2020/Core/cudacore/10.2.89/lib64:$LD_LIBRARY_PATH
module load python/3.8
source ~/cheng/gib/bin/activate
python run_VGPF_pokec.py