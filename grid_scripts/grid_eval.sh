#!/bin/bash

DIRECTORY="results_grid"

for cell in "lstm" "lstmAsum_fn"
do
    for dir in `ls $DIRECTORY/$cell/`
    do
        path="$DIRECTORY/$cell/$dir/model.weights0"
        python lstm.py evaluate -c $cell -m path > $path.output
    done
done
