#!/bin/bash

for d in 0.25 0.5 0.75
do
    for lr in 0.001 0.005 0.01
    do
        for bs in 400 800 1600
        do
            python lstm.py train -d $d -lr $lr -bs $bs
            python lstm.py train -d $d -lr $lr -bs $bs -c lstmAsum_fn
        done
    done
done
