#!/bin/bash
# echo "Bash version ${BASH_VERSION}..."
CMD_PIPELINE="python3 pipeline_resnet_test.py "
CMD_SINGLE="python3 single_resnet_test.py "
CMD_SEQ="python3 sequential_resnet_test.py "
PARTITION_CNT=50
PARTITION_CNT=16
PARTITION_FORMAT="-p "
BATCH_CNT=256
BATCH_CNT=1
BATCH_FORMAT="-b "

# Single execution
# for ((batch=1;batch<=$BATCH_CNT;batch*=2))
# do
#     # echo "Batch : " $batch
#     # for ((partition=1;partition<$PARTITION_CNT;partition++))
#     # do
#         $CMD_SINGLE $BATCH_FORMAT $batch
#     # done
# done

# Sequential execution
# for ((batch=1;batch<=$BATCH_CNT;batch*=2))
# do
#     # echo "Batch : " $batch
#     for ((partition=1;partition<$PARTITION_CNT;partition++))
#     do
#         $CMD_SEQ $PARTITION_FORMAT $partition
#     done
# done


# # # Pipeline execution
for ((batch=1;batch<=$BATCH_CNT;batch*=2))
do
    # echo "Batch : " $batch
    for ((partition=1;partition<$PARTITION_CNT;partition+=1))
    do
        $CMD_PIPELINE $PARTITION_FORMAT $partition $BATCH_FORMAT $batch
    done
done


# echo 1
# python3 pipeline_resnet_test.py -p 1 -b 1
# python3 pipeline_resnet_test.py -p 2 -b 1
# python3 pipeline_resnet_test.py -p 3 -b 1
# python3 pipeline_resnet_test.py -p 4 -b 1
# python3 pipeline_resnet_test.py -p 5 -b 1
# python3 pipeline_resnet_test.py -p 6 -b 1
# python3 pipeline_resnet_test.py -p 7 -b 1
# python3 pipeline_resnet_test.py -p 8 -b 1
# python3 pipeline_resnet_test.py -p 9 -b 1
# python3 pipeline_resnet_test.py -p 10 -b 1
# python3 pipeline_resnet_test.py -p 11 -b 1
# python3 pipeline_resnet_test.py -p 12 -b 1
# python3 pipeline_resnet_test.py -p 13 -b 1
# python3 pipeline_resnet_test.py -p 14 -b 1
# python3 pipeline_resnet_test.py -p 15 -b 1

# echo 2
# python3 pipeline_resnet_test.py -p 1 -b 2
# python3 pipeline_resnet_test.py -p 2 -b 2
# python3 pipeline_resnet_test.py -p 3 -b 2
# python3 pipeline_resnet_test.py -p 4 -b 2
# python3 pipeline_resnet_test.py -p 5 -b 2
# python3 pipeline_resnet_test.py -p 6 -b 2
# python3 pipeline_resnet_test.py -p 7 -b 2
# python3 pipeline_resnet_test.py -p 8 -b 2
# python3 pipeline_resnet_test.py -p 9 -b 2
# python3 pipeline_resnet_test.py -p 10 -b 2
# python3 pipeline_resnet_test.py -p 11 -b 2
# python3 pipeline_resnet_test.py -p 12 -b 2
# python3 pipeline_resnet_test.py -p 13 -b 2
# python3 pipeline_resnet_test.py -p 14 -b 2
# python3 pipeline_resnet_test.py -p 15 -b 2

# echo 4
# python3 pipeline_resnet_test.py -p 1 -b 4
# python3 pipeline_resnet_test.py -p 2 -b 4
# python3 pipeline_resnet_test.py -p 3 -b 4
# python3 pipeline_resnet_test.py -p 4 -b 4
# python3 pipeline_resnet_test.py -p 5 -b 4
# python3 pipeline_resnet_test.py -p 6 -b 4
# python3 pipeline_resnet_test.py -p 7 -b 4
# python3 pipeline_resnet_test.py -p 8 -b 4
# python3 pipeline_resnet_test.py -p 9 -b 4
# python3 pipeline_resnet_test.py -p 10 -b 4
# python3 pipeline_resnet_test.py -p 11 -b 4
# python3 pipeline_resnet_test.py -p 12 -b 4
# python3 pipeline_resnet_test.py -p 13 -b 4
# python3 pipeline_resnet_test.py -p 14 -b 4
# python3 pipeline_resnet_test.py -p 15 -b 4

# echo 8
# python3 pipeline_resnet_test.py -p 1 -b 8
# python3 pipeline_resnet_test.py -p 2 -b 8
# python3 pipeline_resnet_test.py -p 3 -b 8
# python3 pipeline_resnet_test.py -p 4 -b 8
# python3 pipeline_resnet_test.py -p 5 -b 8
# python3 pipeline_resnet_test.py -p 6 -b 8
# python3 pipeline_resnet_test.py -p 7 -b 8
# python3 pipeline_resnet_test.py -p 8 -b 8
# python3 pipeline_resnet_test.py -p 9 -b 8
# python3 pipeline_resnet_test.py -p 10 -b 8
# python3 pipeline_resnet_test.py -p 11 -b 8
# python3 pipeline_resnet_test.py -p 12 -b 8
# python3 pipeline_resnet_test.py -p 13 -b 8
# python3 pipeline_resnet_test.py -p 14 -b 8
# python3 pipeline_resnet_test.py -p 15 -b 8

# echo 16
# python3 pipeline_resnet_test.py -p 1 -b 16
# python3 pipeline_resnet_test.py -p 2 -b 16
# python3 pipeline_resnet_test.py -p 3 -b 16
# python3 pipeline_resnet_test.py -p 4 -b 16
# python3 pipeline_resnet_test.py -p 5 -b 16
# python3 pipeline_resnet_test.py -p 6 -b 16
# python3 pipeline_resnet_test.py -p 7 -b 16
# python3 pipeline_resnet_test.py -p 8 -b 16
# python3 pipeline_resnet_test.py -p 9 -b 16
# python3 pipeline_resnet_test.py -p 10 -b 16
# python3 pipeline_resnet_test.py -p 11 -b 16
# python3 pipeline_resnet_test.py -p 12 -b 16
# python3 pipeline_resnet_test.py -p 13 -b 16
# python3 pipeline_resnet_test.py -p 14 -b 16
# python3 pipeline_resnet_test.py -p 15 -b 16

# echo 32
# python3 pipeline_resnet_test.py -p 1 -b 32
# python3 pipeline_resnet_test.py -p 2 -b 32
# python3 pipeline_resnet_test.py -p 3 -b 32
# python3 pipeline_resnet_test.py -p 4 -b 32
# python3 pipeline_resnet_test.py -p 5 -b 32
# python3 pipeline_resnet_test.py -p 6 -b 32
# python3 pipeline_resnet_test.py -p 7 -b 32
# python3 pipeline_resnet_test.py -p 8 -b 32
# python3 pipeline_resnet_test.py -p 9 -b 32
# python3 pipeline_resnet_test.py -p 10 -b 32
# python3 pipeline_resnet_test.py -p 11 -b 32
# python3 pipeline_resnet_test.py -p 12 -b 32
# python3 pipeline_resnet_test.py -p 13 -b 32
# python3 pipeline_resnet_test.py -p 14 -b 32
# python3 pipeline_resnet_test.py -p 15 -b 32

# echo 64
# python3 pipeline_resnet_test.py -p 1 -b 64
# python3 pipeline_resnet_test.py -p 2 -b 64
# python3 pipeline_resnet_test.py -p 3 -b 64
# python3 pipeline_resnet_test.py -p 4 -b 64
# python3 pipeline_resnet_test.py -p 5 -b 64
# python3 pipeline_resnet_test.py -p 6 -b 64
# python3 pipeline_resnet_test.py -p 7 -b 64
# python3 pipeline_resnet_test.py -p 8 -b 64
# python3 pipeline_resnet_test.py -p 9 -b 64
# python3 pipeline_resnet_test.py -p 10 -b 64
# python3 pipeline_resnet_test.py -p 11 -b 64
# python3 pipeline_resnet_test.py -p 12 -b 64
# python3 pipeline_resnet_test.py -p 13 -b 64
# python3 pipeline_resnet_test.py -p 14 -b 64
# python3 pipeline_resnet_test.py -p 15 -b 64

# echo 128
# python3 pipeline_resnet_test.py -p 1 -b 128
# python3 pipeline_resnet_test.py -p 2 -b 128
# python3 pipeline_resnet_test.py -p 3 -b 128
# python3 pipeline_resnet_test.py -p 4 -b 128
# python3 pipeline_resnet_test.py -p 5 -b 128
# python3 pipeline_resnet_test.py -p 6 -b 128
# python3 pipeline_resnet_test.py -p 7 -b 128
# python3 pipeline_resnet_test.py -p 8 -b 128
# python3 pipeline_resnet_test.py -p 9 -b 128
# python3 pipeline_resnet_test.py -p 10 -b 128
# python3 pipeline_resnet_test.py -p 11 -b 128
# python3 pipeline_resnet_test.py -p 12 -b 128
# python3 pipeline_resnet_test.py -p 13 -b 128
# python3 pipeline_resnet_test.py -p 14 -b 128
# python3 pipeline_resnet_test.py -p 15 -b 128

# echo 256
# python3 pipeline_resnet_test.py -p 1 -b 256
# python3 pipeline_resnet_test.py -p 2 -b 256
# python3 pipeline_resnet_test.py -p 3 -b 256
# python3 pipeline_resnet_test.py -p 4 -b 256
# python3 pipeline_resnet_test.py -p 5 -b 256
# python3 pipeline_resnet_test.py -p 6 -b 256
# python3 pipeline_resnet_test.py -p 7 -b 256
# python3 pipeline_resnet_test.py -p 8 -b 256
# python3 pipeline_resnet_test.py -p 9 -b 256
# python3 pipeline_resnet_test.py -p 10 -b 256
# python3 pipeline_resnet_test.py -p 11 -b 256
# python3 pipeline_resnet_test.py -p 12 -b 256
# python3 pipeline_resnet_test.py -p 13 -b 256
# python3 pipeline_resnet_test.py -p 14 -b 256
# python3 pipeline_resnet_test.py -p 15 -b 256

# echo 512
# python3 pipeline_resnet_test.py -p 1 -b 512
# python3 pipeline_resnet_test.py -p 2 -b 512
# python3 pipeline_resnet_test.py -p 3 -b 512
# python3 pipeline_resnet_test.py -p 4 -b 512
# python3 pipeline_resnet_test.py -p 5 -b 512
# python3 pipeline_resnet_test.py -p 6 -b 512
# python3 pipeline_resnet_test.py -p 7 -b 512
# python3 pipeline_resnet_test.py -p 8 -b 512
# python3 pipeline_resnet_test.py -p 9 -b 512
# python3 pipeline_resnet_test.py -p 10 -b 512
# python3 pipeline_resnet_test.py -p 11 -b 512
# python3 pipeline_resnet_test.py -p 12 -b 512
# python3 pipeline_resnet_test.py -p 13 -b 512
# python3 pipeline_resnet_test.py -p 14 -b 512
# python3 pipeline_resnet_test.py -p 15 -b 512
