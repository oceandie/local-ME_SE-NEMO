#!/bin/bash -l

#SBATCH --qos=normal                
#SBATCH --mem=100000                    
#SBATCH --ntasks=8                   
#SBATCH --output=hpge_2D_mean.out       
#SBATCH --time=360                    

python plot_mean-vel_timeseries.py 
