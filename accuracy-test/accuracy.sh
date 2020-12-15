#!/bin/bash

# dataset=$1
# model=$2

for arg in "$@"; do
  if [ -z "$dataset" ]; then
    dataset=$arg
  elif [ -z "$model" ]; then
    model=$arg
  else
    classes+=("$arg")
  fi
done

classes_str=""
for class in ${classes[@]}; do
  classes_str="${classes_str}${class},"
done

cd ../ground_truth/${dataset}/${model} 
results=`ls -1 | grep .pkl`
cd -

results_path=${dataset}/results_${model}
mkdir -p ${results_path}
csvfile=${results_path}/all_results.csv
header="cam,chunk,ts,date,hour,min,${classes_str}mAP"
echo $header > $csvfile

testbin="python ../accuracy-metrics/pascalvoc.py"
for res in `echo "${results}"`; do

  if [ ! -f "../ground_truth/${dataset}/ref/${res}" ]; then
    echo "Skipping ${res} because it has no groundtruth results"
    continue
  fi

  timestamp=`echo ${res} | cut -d'.' -f1`
  cam=`echo ${res} | cut -d'.' -f2`
  chunk=`echo ${res} | cut -d'.' -f3`
  pickle="${timestamp}.${cam}.${chunk}"
  if [ "$chunk" == "pkl" ]; then
    chunk="0"
    pickle="${timestamp}.${cam}"
  fi
  path="../accuracy-test/${dataset}/${model}/${pickle}"
  
  # 1. Generate groundtruth files if don't exist
  cd ../post
  # if [ ! -d ${path}/groundtruths ]; then
  mkdir -p ${path}/groundtruths
  pid=0
  if [ ! -f ${path}/groundtruths/${pikle}_0.txt ]; then
    python generate_detection_files.py ../ground_truth/${dataset}/ref/${res} --output ${path} --ground-truth -t 0.5 &
    pid=$!
  fi

  # 2. Generate detection files if don't exist
  # if [ ! -d ${path}/detections ]; then
  mkdir -p ${path}/detections
  if [ ! -f ${path}/detections/${pikle}_0.txt ]; then
    python generate_detection_files.py ../ground_truth/${dataset}/edge/${res} --output ${path}
  fi

  if [ ! "$pid" == "0" ]; then
    wait $pid
  fi

  cd -
  # 3. Compute VOC metrics
  if [ ! -f ${path}/results/results.txt ]; then
    rm -rf ${path}/results
    mkdir -p ${path}/results
    $testbin -t 0.3 -det ${path}/detections -gt ${path}/groundtruths -gtformat xyrb -detformat xyrb -sp ${path}/results -np --classes ${classes_str} 
  fi

  cp ${path}/results/results.txt ${results_path}/${pickle}_results.txt
  cp ${path}/results/car.png ${results_path}/${pickle}_car.png
  cp ${path}/results/person.png ${results_path}/${pickle}_person.png

  resfile="${results_path}/${pickle}_results.txt"
  date=`echo ${timestamp} | cut -b5-8`
  hour=`echo ${timestamp} | cut -b9-10`
  min=`echo ${timestamp} | cut -b11-12`
  row="${cam},${chunk},${timestamp},${date},${hour},${min},"
  for class in ${classes[@]}; do
    avg_pr=`grep "Class: ${class}" ${resfile} -A1 | grep "AP" | cut -d' ' -f2 | sed 's/%//'`
    row="${row}${avg_pr},"
  done

  mAP=`grep "mAP:" ${resfile} -A1 | cut -d' ' -f2 | sed 's/%//'`
  row="${row}${mAP}"
  echo $row
  echo $row >> $csvfile
done

# cd $dataset
# results=`ls -1 | grep -E "[0-9]+"`
# echo $header
# for res in `echo "${results}"`; do
  # echo $res
# done
