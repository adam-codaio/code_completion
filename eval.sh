#!/bin/bash

for i in 9 10 11
do
    python lstm.py evaluate -m $1$i >> output
done
