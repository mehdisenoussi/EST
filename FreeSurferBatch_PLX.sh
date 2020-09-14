#!/bin/bash
#PBS -l nodes=1:ppn=4
#PBS -N FreeSurfer_per_subject-time
#PBS -l walltime=14:00:00
#PBS -m abe
#PBS -j oe

#== LOAD LATEST VERSION OF FREESURFER ==#
module load FreeSurfer

#== INITIALIZE FREESURFER ==#
# To be adapted:
#  - HOMEDIR	   Run the bash from the home directory or change the path in the script
#  - FS_LICENSE	   Change to path so it points to the location of the license file 	
#  - SUBJECTS_DIR  Change to desired output folder location
#  - INPUT_DIR	   Change to folder containing all participant .nii files	 
		
export HOMEDIR=$(pwd) # Be sure to run this shell from home directory
export FS_LICENSE=$HOMEDIR/license.txt # Change path to point to license file
export SUBJECTS_DIR=$HOMEDIR/data/PLX # Change path to point to input folder containing the .nii of all subjects
export INPUT_DIR=$HOMEDIR/FS_Data/input_PLX
source $FREESURFER_HOME/SetUpFreeSurfer.sh 

#== READ IN SUBJECT NAMES AND .NII LOCATION ==#
dos2unix $HOMEDIR/inputs_per-subject-time_PLX.txt
export SUBJ=`sed -n "${PBS_ARRAYID}p" $HOMEDIR/inputs_per-subject-time_CO.txt` #This extracts the subject names from the .txt file

#== RUN FREESURFER ==#
recon-all -i $INPUT_DIR/$SUBJ.nii -subject $SUBJ -all -notal-check -qcache
