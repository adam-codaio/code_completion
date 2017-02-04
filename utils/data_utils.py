import json
import os
import time

"""
Largely inspired by code from cs224n PSET 2
"""

I_PREFIX = '<i>:'
T_PREFIX = '<t>:'
V_PREFIX = '<v>:'
C_PREFIX = '<c>:'

class Config(object):
    data_path = './data'
    reduced_train = 500
    reduced_dev = 100
    reduced_test = 100
    train_file = 'programs_training.json'
    dev_file = 'programs_dev.json'
    test_file = 'programs_test.json'

class CodeCompleter(object):
    def __init__(self, dataset):
        for ex in dataset:
            for node in ex:
               pass       
   
    def vectorize(self, examples):
        return examples

    def create_instances(self, examples):
        return examples

def read_data(infile, reduced=False, num_examples=None):
    examples = []
 
    with open(infile) as f:
        for line in f:
            data = json.loads(line)
            examples.append(data)
            if reduced:
                num_examples -= 1
                if num_examples == 0:
                    break

    return examples

def load_and_preprocess_data(reduced=True):
    config = Config()
    
    print "Loading data...",
    start = time.time()

    train_set = read_data(os.path.join(config.data_path, config.train_file),
                          reduced, config.reduced_train)
    dev_set = read_data(os.path.join(config.data_path, config.dev_file),
                        reduced, config.reduced_dev)
    test_set = read_data(os.path.join(config.data_path, config.test_file),
                         reduced, config.reduced_test)
    print "took {:.2f} seconds".format(time.time() - start)
    
    print "Building Code Completer...",
    start = time.time()
    code_comp = CodeCompleter(train_set)
    print "took {:.2f} seconds".format(time.time() - start)

    print "Vectorizing data...",
    start = time.time()
    train_set = code_comp.vectorize(train_set)
    dev_set = code_comp.vectorize(dev_set)
    test_set = code_comp.vectorize(test_set)
    print "took {:.2f} seconds".format(time.time() - start)

    print "Preprocessing training data"
    train_examples = code_comp.create_instances(train_set)
    
    return code_comp, train_examples, dev_set, test_set
