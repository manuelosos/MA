#!/bin/bash
#SBATCH --job-name=ma-simulation
#SBATCH --time=01:00:00
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --mail-user=manuel.urban@uni-bayreuth.de
#SBATCH --chdir=/scratch/bt705242/ma
#SBATCH --error=%x_%j.err #Fehlerausgabedatei %x:= jobname,%j:=jobid
#SBATCH --output=%x_%j.out
#SBATCH --ntasks=16
#SBATCH --chdir=/scratch/bt705242/ma/MA/code/simulation/

export MYRUNFILES="runfiles/test2.json"
export MYRESULTS="results"

module load julia/1.6.3

julia simulation.jl



echo "Operation nach ${rt} Sekunden beendet"