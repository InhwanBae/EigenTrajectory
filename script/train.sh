#!/bin/bash
echo "Start training task queues"

# Hyperparameters
dataset_array=("eth" "hotel" "univ" "zara1" "zara2")
device_id_array=(0 1 2 3 4)
tag="EigenTrajectory"
config_path="./config/"
config_prefix="eigentrajectory"
baseline="sgcn"

# Arguments
while getopts t:b:c:p:d:i: flag
do
  case "${flag}" in
    t) tag=${OPTARG};;
    b) baseline=${OPTARG};;
    c) config_path=${OPTARG};;
    p) config_prefix=${OPTARG};;
    d) dataset_array=(${OPTARG});;
    i) device_id_array=(${OPTARG});;
    *) echo "usage: $0 [-t TAG] [-b BASELINE] [-c CONFIG_PATH] [-d \"eth hotel univ zara1 zara2\"] [-i \"0 1 2 3 4\"]" >&2
      exit 1 ;;
  esac
done

if [ ${#dataset_array[@]} -ne ${#device_id_array[@]} ]
then
    printf "Arrays must all be same length. "
    printf "len(dataset_array)=${#dataset_array[@]} and len(device_id_array)=${#device_id_array[@]}\n"
    exit 1
fi

# Signal handler
pid_array=()

sighdl ()
{
  echo "Kill training processes"
  for (( i=0; i<${#dataset_array[@]}; i++ ))
  do
    kill ${pid_array[$i]}
  done
  echo "Done."
  exit 0
}

trap sighdl SIGINT SIGTERM

# Start training tasks
for (( i=0; i<${#dataset_array[@]}; i++ ))
do
  printf "Evaluate ${dataset_array[$i]}"
  CUDA_VISIBLE_DEVICES=${device_id_array[$i]} python3 trainval.py \
  --cfg "${config_path}""${config_prefix}"-"${baseline}"-"${dataset_array[$i]}".json \
  --tag "${tag}" --gpu_id ${device_id_array[$i]} &
  pid_array[$i]=$!
  printf " job ${#pid_array[@]} pid ${pid_array[$i]}\n"
done

for (( i=0; i<${#dataset_array[@]}; i++ ))
do
  wait ${pid_array[$i]}
done

echo "Done."
