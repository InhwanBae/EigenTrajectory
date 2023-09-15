#!/bin/bash

baseline_array=("STGCNN" "SGCN" "PECNet" "AgentFormer" "LBEBM" "DMRGCN" "GPGraph-STGCNN" "GPGraph-SGCN" "Graph-TERN" "Implicit")

for (( i=0; i<${#baseline_array[@]}; i++ ))
do
  echo "Download pre-trained model with ${baseline_array[$i]} baseline."
  wget -O ${baseline_array[$i]}.zip https://github.com/InhwanBae/EigenTrajectory/releases/download/v1.0/EigenTrajectory-${baseline_array[$i]}-pretrained.zip
  unzip -q ${baseline_array[$i]}.zip
  rm -rf ${baseline_array[$i]}.zip
done

echo "Done."
