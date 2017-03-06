# Code Completion
cs224n Winter 2017 final project. Code completion using deep learning.

## Data
Run the steps below on corn (or other remote server). This will take a while.
```
wget http://files.srl.inf.ethz.ch/data/js_dataset.tar.gz
mkdir data
tar xzvf js_dataset.tar.gz -C data
mv js_dataset.tar.gz data/
cd data
tar xzvf data.tar.gz

//Splits the eval set into an eval and dev set
cd utils
python eval_split.py
```

##Train Glove Vectors
Step 1: Clone Glove (clone this repository outside of your code_completion directory: https://github.com/stanfordnlp/GloVe)
```
$ git clone http://github.com/stanfordnlp/glove
$ cd glove && make
```

Step 2: Build Word Corpus 
```
$ python build_glove_corpus.py
```

Step 3: Train Glove
```
$ ./trainGlove.sh
```
