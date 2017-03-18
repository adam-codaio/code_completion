#!/bin/bash

FROMDIRECTORY="results"
TODIRECTORY="results_grid"

for cell in "lstm" "lstmAsum_fn"
do
    for dir in `ls $FROMDIRECTORY/$cell | tail -27`
    do
        mv $FROMDIRECTORY/$cell/$dir $TODIRECTORY/$cell/
    done
done
