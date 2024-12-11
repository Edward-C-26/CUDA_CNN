#!/bin/bash
#SBATCH --job-name="Milestone3"
#SBATCH --output="Milestone3.out"
#SBATCH --error="Milestone3.err"
#SBATCH --partition=gpuA40x4
#SBATCH --mem=12G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --constraint="projects"
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest
#SBATCH --account=bche-delta-gpu

#SBATCH --mail-type=END
#SBATCH --mail-user=ec34@illinois.edu

#SBATCH -t 00:25:00

module reset
module load cuda

echo -e "job $SLURM_JOBID is starting on `hostname`\n\n"

srun cuda-memcheck ./m3_req1 100000 > m3_req1_10000.out

# Generate nsys profile information, output will be in profile.out
# srun nsys profile -o profiled_file3_req0 ./m3  10000 > profile_m3_req0.out
# srun nsys profile --stats=true -o profiled_file3_req0 ./m3  10000 > profile_m3_req0.out

# srun nsys profile -o profiled_file3_op0 --stats=true ./m3_op0 10000 > nsys3_op0.out
# srun nsys profile -o profiled_file3_op1 --stats=true ./m3_op1 10000 > nsys3_op1.out
# srun nsys profile -o profiled_file3_op2 --stats=true ./m3_op2 10000 > nsys3_op2.out
# srun nsys profile -o profiled_file3_op4 --stats=true ./m3_op4 10000 > nsys3_op4.out
# srun nsys profile -o profiled_file3_req0 --stats=true ./m3_req0 10000 > nsys3_req0.out
# srun nsys profile -o profiled_file3_req1 --stats=true ./m3_req1 10000 > nsys3_req1.out
# srun nsys profile -o profiled_file3_req2 --stats=true ./m3_req2 10000 > nsys3_req2.out



# Generate analysis_file.ncu-rep with a detailed profiling result
# srun ncu --set full -f -o analysis_file3_op0 ./m3_op0 10000 > ncu3_op0.out
# srun ncu --set full -f -o analysis_file3_op1 ./m3_op1 10000 > ncu3_op1.out
# srun ncu --set full -f -o analysis_file3_op2 ./m3_op2 10000 > ncu3_op2.out
# srun ncu --set full -f -o analysis_file3_op4 ./m3_op4 10000 > ncu3_op4.out
# srun ncu --set full -f -o analysis_file3_req0 ./m3_req0 10000 > ncu3_req0.out
# srun ncu --set full -f -o analysis_file3_req1 ./m3_req1 10000 > ncu3_req1.out
# srun ncu --set full -f -o analysis_file3_req2 ./m3_req2 10000 > ncu3_req2.out



# { time srun ./m3_req0 100 > m3_req0_100.out; } 2> m3_req0_100.out
# { time srun ./m3_req0 1000 > m3_req0_1000.out; } 2> m3_req0_1000.out
# { time srun ./m3_req0 10000 > m3_req0_10000.out; } 2> m3_req0_10000.out

# { time srun ./m3_req1 100 > m3_req1_100.out; } 2> m3_req1_100.out
# { time srun ./m3_req1 1000 > m3_req1_1000.out; } 2> m3_req1_1000.out
# { time srun ./m3_req1 10000 > m3_req1_10000.out; } 2> m3_req1_10000.out

# { time srun ./m3_req2 100 > m3_req2_100.out; } 2> m3_req2_100.out
# { time srun ./m3_req2 1000 > m3_req2_1000.out; } 2> m3_req2_1000.out
# { time srun ./m3_req2 10000 > m3_req2_10000.out; } 2> m3_req2_10000.out

# { time srun ./m3_op0 100 > m3_op0_100.out; } 2> m3_op0_100.out
# { time srun ./m3_op0 1000 > m3_op0_1000.out; } 2> m3_op0_1000.out
# { time srun ./m3_op0 10000 > m3_op0_10000.out; } 2> m3_op0_10000.out

# { time srun ./m3_op1 100 > m3_op1_100.out; } 2> m3_op1_100.out
# { time srun ./m3_op1 1000 > m3_op1_1000.out; } 2> m3_op1_1000.out
# { time srun ./m3_op1 10000 > m3_op1_10000.out; } 2> m3_op1_10000.out

# { time srun ./m3_op2 100 > m3_op2_100.out; } 2> m3_op2_100.out
# { time srun ./m3_op2 1000 > m3_op2_1000.out; } 2> m3_op2_1000.out
# { time srun ./m3_op2 10000 > m3_op2_10000.out; } 2> m3_op2_10000.out

# { time srun ./m3_op4 100 > m3_op4_100.out; } 2> m3_op4_100.out
# { time srun ./m3_op4 1000 > m3_op4_1000.out; } 2> m3_op4_1000.out
# { time srun ./m3_op4 10000 > m3_op4_10000.out; } 2> m3_op4_10000.out

#,perf,nvperf