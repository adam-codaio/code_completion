#!/bin/bash

for i in 0 1 2 3 4 5 6 7 8 9 10 11
do
    python lstm.py evaluate -m $1$i >> output
done
