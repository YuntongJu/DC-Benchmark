#!/bin/bash
dataset=${dataset:-cifar10}
method=${method:-"DC"}
aug=${method:-"random_aug"}
gpu=${gpu:-"auto"}
ipc=${ipc:-1}
model=${model:"convnet"}

while [ $# -gt 0 ]; do
    if [[ $1 == *"--"* ]]; then
        param="${1/--/}"
        declare $param="$2"
    fi
    shift
done

echo 'method:' $method 'dataset:' $dataset 'aug:' $aug
echo 'gpu:' $gpu

cd ../

if [[ $method == "DC" ]]; then
    python distilled_results/DC/dc_evaluator.py \
        --dataset $dataset \
        --aug $aug \
        --gpu $gpu \
        --ipc $ipc \
        --model $model
elif [[ $method == "DSA" ]]; then
    python distilled_results/DC/dsa_evaluator.py \
        --dataset $dataset \
        --aug $aug \
        --gpu $gpu \
        --ipc $ipc \
        --model $model
else
  STATEMENTS3
fi
    # --fast --expid_tag debug \

## bash darts-201.sh