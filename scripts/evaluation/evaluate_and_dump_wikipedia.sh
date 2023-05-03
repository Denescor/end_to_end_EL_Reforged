#!/bin/bash
# script bash expérience : evaluate kolitsas trained on wikipedia on DBpedia and TRFR2016

#SBATCH --qos=qos_gpu-t3
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:4
#SBATCH -C v100
#SBATCH --nodes=1
#SBATCH -A gcp@v100

#SBATCH --array=0-2

#SBATCH --time=1:00:00

#SBATCH --output=%j.out
#SBATCH --error=%j.err

#SBATCH --job-name=EVAL_K
#SBATCH --signal=B:USR1@120

# Load Environnement
module load python/3.7.5
#module load cpuarch/amd
#module load tensorflow-gpu/py3/2.8.0
export PYTHONUSERBASE=$WORK/.local_endtoendEL #_A100
export PATH=$PYTHONUSERBASE/bin:$PATH

ARGUMENT=$1 # File with specific arguments
MODE=$2 # mode : just eval or dump outputs

# Verif arguments
if [ "$ARGUMENT" == "" ]; then
    echo "ARGUMENT have to be set"
    exit
fi

if [ "$MODE" == "" ]; then
    MODE="EVAL"
    echo "Default mode : EVAL"
elif [ "$MODE" == "EVAL" ] || [ "$MODE" == "DUMP" ];then
    echo "Selected mode : $MODE"
else
    echo "mode must be 'EVAL' or 'DUMP'"
    exit
fi

IFS=$'\n' read -d '' -r -a arguments < ./arguments/${ARGUMENT} #

DIR="/mnt/beegfs/home/carpentier/endtoendEL/end2end_neural_el-master/data/tfrecords"
FOLDER_OPTION="--training_name="
GENERAL_OPTION="--no_print_predictions --ed_datasets=  --el_val_datasets=0 --ed_val_datasets=0 --all_spans_training=True "
EXP_FOLDER="model_corefmerge"

function restart { echo "Restarting required" >> $LOGSTDOUT ; scontrol requeue $SLURM_JOB_ID ; echo "Scheduled job" >> $LOGSTDOUT ; }

function ignore { echo "SIGTERM ignored" >> $LOG_STDOUT ; }
trap restart USR1
trap ignore TERM

date
echo "#### START RUN ####"
echo "name : $MODE KOLITSAS FOR $ARGUMENT"
echo "Job ID : $SLURM_JOB_ID"
cd ../..

args=()

for s in "${arguments[@]}"
do
    IFS='|' read -ra options <<< "${s}"
    EXP_NAME="--experiment_name=${options[0]} "
    DIR_OPTION="${EXP_NAME}${FOLDER_OPTION}${options[1]}/${EXP_FOLDER}"
    FINAL_ARGS="${DIR_OPTION} ${GENERAL_OPTION} ${options[2]}"
    args+=("$FINAL_ARGS") # Generate args for all expériences
done

set -x

if [ "$MODE" == "EVAL" ]; then
    python -m model.evaluate ${args[${SLURM_ARRAY_TASK_ID}]}
else # MODE == "DUMP"
    python -m model.usemodel ${args[${SLURM_ARRAY_TASK_ID}]}
fi

echo "FIN"
date
