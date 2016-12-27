#!/bin/bash

# Base conv depth
for i in 5
do
  # FCL depth
  for j in 220
  do
    # LR init
    for k in 0.013
    do
      # LR decay
      for m in 0.99
      do
        # Dropout keep prob
        for n in 0.5
        do
          # L2 reg
          for p in 0.0005
          do
            # Batch size
            for q in 75
            do
              # Run several separate training sessions.
              for r in `seq 1 20`
              do
                # Remove existing checkpoint and start new training session.
                rm model/jz-cnn/checkpoint
                echo "TRAINING SESSION STARTING: $r"
                python TrainJzCNN.py $i $j $k $m $n $p $q 2>&1 | tee model/jz-cnn/train-$i-$j-$k-$m-$n-$p-$q-$r.log

                # Save best model and remove other.
                mv model/jz-cnn/train-vars-best model/jz-cnn/train-vars-best-$i-$j-$k-$m-$n-$p-$q-$r
                rm model/jz-cnn/train-vars

                # Backup train log tails and error stats.
                ./write_backup_jz.sh "DS: 7200x1440, KS: 3333, CLD: $i, FCL: $j, LRInit: $k, LRDecay: $m, DOKeepProb: $n, L2Reg: $p, BS: $q"

                # Move log into subdirectory to prevent reprocessing.
                mv model/jz-cnn/train-*.log model/jz-cnn/logs/
              done
            done
          done
        done
      done
    done
  done
done

