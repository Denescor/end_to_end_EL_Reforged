#!/bin/bash
# JID_JOB0=`sbatch  lauch_exp_fr_09.sh | cut -d " " -f 4` #initial training
JID_JOB1=`sbatch lauch_exp_fr_continue.sh | cut -d " " -f 4` #continue 1 time
JID_JOB2=`sbatch --dependency=afterok:$JID_JOB1 lauch_exp_fr_continue.sh | cut -d " " -f 4` #continue 2 time
JID_JOB3=`sbatch --dependency=afterok:$JID_JOB2 lauch_exp_fr_continue.sh | cut -d " " -f 4` #continue 3 time
sbatch --dependency=afterok:$JID_JOB3 lauch_exp_fr_continue.sh #last time
