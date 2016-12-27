#!/bin/bash

# Cycle through all 4 training sets 3 times.
for j in `seq 1`
do
    for i in `seq 1 4`
    do
        echo "TRAINING SESSION STARTING: $j $i"
        python TrainMainCNN.py $i $1 $2 $3 $4 $5 $6 $7
    done
done

