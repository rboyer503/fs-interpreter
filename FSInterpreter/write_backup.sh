#!/bin/bash

BACKUPFILENAME=model/main-cnn/logs/train_notes

echo "$1" >> ${BACKUPFILENAME}

echo "Train log tails:" >> ${BACKUPFILENAME}
tail -n 8 model/main-cnn/train-*.log >> ${BACKUPFILENAME}

echo >> ${BACKUPFILENAME}
echo "Test error stats:" >> ${BACKUPFILENAME}
echo -n "# test error 1.x: " >> ${BACKUPFILENAME}
grep 'Test error: 1\.' model/main-cnn/train-*.log | wc -l >> ${BACKUPFILENAME}
echo -n "# test error 2.x: " >> ${BACKUPFILENAME}
grep 'Test error: 2\.' model/main-cnn/train-*.log | wc -l >> ${BACKUPFILENAME}
echo -n "# test error 3.x: " >> ${BACKUPFILENAME}
grep 'Test error: 3\.' model/main-cnn/train-*.log | wc -l >> ${BACKUPFILENAME}
echo "----------" >> ${BACKUPFILENAME}
echo >> ${BACKUPFILENAME}

