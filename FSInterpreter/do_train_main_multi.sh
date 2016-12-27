#!/bin/bash

# Base conv depth
for i in 10
do
  # FCL depth
  for j in 100
  do
    # LR init
    for k in 0.015
    do
      # LR decay
      for m in 0.95
      do
        # Dropout keep prob
        for n in 0.5
        do
          # L2 reg
          for p in 0.0005
          do
            # Batch size
            for q in 100
            do
              # Run several separate training sessions.
              for r in `seq 1 20`
              do
                # Remove existing checkpoint and start new training session.
                rm model/main-cnn/checkpoint
                ./do_train_main.sh $i $j $k $m $n $p $q 2>&1 | tee model/main-cnn/train-$i-$j-$k-$m-$n-$p-$q-$r.log

                # Save best model and remove other.
                mv model/main-cnn/train-vars-best model/main-cnn/train-vars-best-$i-$j-$k-$m-$n-$p-$q-$r
                rm model/main-cnn/train-vars

                # Backup train log tails and error stats.
                ./write_backup.sh "DS: 142500x3192, KS: 5555, CLD: $i, FCL: $j, LRInit: $k, LRDecay: $m, DOKeepProb: $n, L2Reg: $p, BS: $q"

                # Move log into subdirectory to prevent reprocessing.
                mv model/main-cnn/train-*.log model/main-cnn/logs/
              done
            done
          done
        done
      done
    done
  done
done

