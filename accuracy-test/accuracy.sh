#!/bin/bash

dataset=$1
model=$2

classes=(car person)

classes_str=""
for class in ${classes[@]}; do
  classes_str="${classes_str}${class},"
done
# cd ../ground_truth/${dataset}/${model} 
# results=`ls -1 | grep .pkl | cut -d'.' -f1`
# cd -
# 
# testbin="python ../accuracy-metrics/pascalvoc.py"
# for res in `echo "${results}"`; do
#   path="../accuracy-test/${dataset}/${res}"
# 
#   # 1. Generate groundtruth files if don't exist
#   cd ../post
#   # if [ ! -d ${path}/groundtruths ]; then
#     mkdir -p ${path}/groundtruths
#     python generate_detection_files.py ../ground_truth/${dataset}/ref/$res.bcn.pkl --output ${path} --ground-truth -t 0.5
#   # fi
# 
#   # 2. Generate detection files if don't exist
#   # if [ ! -d ${path}/detections ]; then
#     mkdir -p ${path}/detections
#     python generate_detection_files.py ../ground_truth/${dataset}/edge/$res.bcn.pkl --output ${path}
#   # fi
# 
#   cd -
#   # 3. Compute VOC metrics
#   if [ -d ${path}/results ]; then
#     rm -rf ${path}/results
#   fi
#   mkdir -p ${path}/results
#   $testbin -t 0.3 -det ${path}/detections -gt ${path}/groundtruths -gtformat xyrb -detformat xyrb -sp ${path}/results -np --classes car,person
# 
# done

cd $dataset
results=`ls -1 | grep -E "[0-9]+"`
mkdir -p results
csvfile=results/all_results.csv
header="ts,date,hour,min,${classes_str}mAP"
echo $header
echo $header > $csvfile
for res in `echo "${results}"`; do
  # echo $res
  cp $res/results/results.txt results/${res}_results.txt
  cp $res/results/car.png results/${res}_car.png
  cp $res/results/person.png results/${res}_person.png

  resfile="results/${res}_results.txt"
  date=`echo ${res} | cut -b5-8`
  hour=`echo ${res} | cut -b9-10`
  min=`echo ${res} | cut -b11-12`
  row="${res},${date},${hour},${min},"
  for class in ${classes[@]}; do
    avg_pr=`grep "Class: ${class}" ${resfile} -A1 | grep "AP" | cut -d' ' -f2`
    row="${row}${avg_pr},"
  done

  mAP=`grep "mAP:" ${resfile} -A1 | cut -d' ' -f2`
  row="${row}${mAP}"
  echo $row
  echo $row >> $csvfile
done
