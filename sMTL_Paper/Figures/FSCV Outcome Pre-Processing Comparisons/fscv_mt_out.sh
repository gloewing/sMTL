#!/bin/bash
#SBATCH --job-name sparseRegMT      # Set a name for your job. This is especially useful if you
#SBATCH --partition shared     # Slurm partition to use
#SBATCH -c 3        # Number of tasks to run
#SBATCH --time 0-6:00       # Wall time limit in D-HH:MM
#SBATCH --mem 2500     # Memory limit for each tasks (in MB) # 1500
#SBATCH -o /n/home12/gloewinger/error/output_fscvMT.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e /n/home12/gloewinger/error/errors_fscvMT.err  # File to which STDERR will be written, %j inserts jobid
#SBATCH --array 1-30 # number should match number $1 below

module load gcc/7.1.0-fasrc01 R/3.5.1-fasrc02
export PATH=/n/home12/gloewinger/apps/gurobi811/linux64/bin:$PATH
export LD_LIBRARY_PATH=/n/home12/gloewinger/apps/glpk-4.65/lib:/n/home12/gloewinger/apps/gurobi811/linux64/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=/n/home12/gloewinger/apps/glpk-4.65/lib:$LIBRARY_PATH
export CPATH=/n/home12/gloewinger/apps/glpk-4.65/include:$PATH
export PATH=~/mosek/9.1/tools/platform/linux64x86/bin:$PATH
export R_LIBS_USER=$HOME/apps/R_3.5.1:$R_LIBS_USER
module load R/3.5.1-fasrc01

Rscript '/n/home12/gloewinger/fscv_multiStudy_IHT_MT_outcomeTest_cf.R' $1 $2 # first is number of CV splits for testing and second is which subset of IDPs to use as tasks (DMN) (can be 1,2 or 3). $1 should match the array index (line ~9)
