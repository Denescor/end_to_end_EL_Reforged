#!/bin/bash
# script bash expérience : expériences transferts learning Kolitsas corefmerge - transfert + baseline sans transfert

#SBATCH --qos=qos_gpu-t4
#SBATCH --cpus-per-task=60
#SBATCH --gres=gpu:4
#SBATCH -C v100-32g
#SBATCH --nodes=1
#SBATCH -A gcp@v100

#SBATCH --time=98:00:00

#SBATCH --array=0-1

#SBATCH --output=%j.out
#SBATCH --error=%j.err

#SBATCH --job-name=tfr_Kolitsas
#SBATCH --signal=B:USR1@120

# Load Environnement
module load python/3.7.5
export PYTHONUSERBASE=$WORK/.local_endtoendEL
export PATH=$PYTHONUSERBASE/bin:$PATH

ARGUMENT=$1
if [ "$ARGUMENT" == "" ]; then
    echo "ARGUMENT have to be set"
    exit
fi

IFS=$'\n' read -d '' -r -a arguments < arguments/${ARGUMENT} #

DIR="/mnt/beegfs/home/carpentier/endtoendEL/end2end_neural_el-master/data/tfrecords"
FOLDER_OPTION="--training_name=" #${DIR}"
GENERAL_OPTION="--all_spans_training=True --fast_evaluation=True --evaluation_minutes=30 --continue_training=True"
EXP_FOLDER="model_corefmerge"

function restart { echo "Restarting required" >> $LOGSTDOUT ; scontrol requeue $SLURM_JOB_ID ; echo "Scheduled job" >> $LOGSTDOUT ; }

function ignore { echo "SIGTERM ignored" >> $LOG_STDOUT ; }
trap restart USR1
trap ignore TERM

date
echo "#### START RUN ####"
echo "name : EXP ${ARGUMENT}"
echo "Job ID : $SLURM_JOB_ID"
cd ../..

args=()

for s in "${arguments[@]}"
do
    IFS='|' read -ra options <<< "${s}"
    EXP_NAME="--experiment_name=${options[0]} "
    DIR_OPTION="${EXP_NAME}${FOLDER_OPTION}${options[1]}/${EXP_FOLDER}"
    args+=("${DIR_OPTION} ${GENERAL_OPTION} ${options[2]}") # Generate args for all expériences
done

set -x

echo "${#arguments[@]}"
printf "%s\n" "${args[@]}"

python -m model.train ${args[${SLURM_ARRAY_TASK_ID}]}

echo "FIN"
date
