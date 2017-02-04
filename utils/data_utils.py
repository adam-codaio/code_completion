import json
import os
import time

"""
Largely inspired by code from cs224n PSET 2 for the 
dependency parser
"""

I_PREFIX = '<i>:'
T_PREFIX = '<t>:'
V_PREFIX = '<v>:'
C_PREFIX = '<c>:'

NON_ASCII = '<NON_ASCII>'
I_UNK = '<I_UNK>'
T_UNK = '<T_UNK>'
V_UNK = '<V_UNK>'
C_UNK = '<C_UNK>'
V_NULL = '<V_NULL>'
C_NULL = '<C_NULL>'

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
        tok2id = {};
        for ex in dataset:
            for node in ex:
               if node == 0:
                   continue
               i = str(node['id'])
               if I_PREFIX + i not in tok2id:
                   tok2id[I_PREFIX + i] = len(tok2id)
               t = node['type']
               if T_PREFIX + t not in tok2id:
                   tok2id[T_PREFIX + t] = len(tok2id)
               v = node.get('value', False)
               if v:
                   try:
                       v = str(v)
                   except:
                       v = NON_ASCII
                   if V_PREFIX + v not in tok2id:
                       tok2id[V_PREFIX + v] = len(tok2id)
               c = node.get('children', False)
               if c:
                   c = str(c)
                   if C_PREFIX + c not in tok2id:
                       tok2id[C_PREFIX + c] = len(tok2id)

        tok2id[I_UNK] = self.I_UNK = len(tok2id)
        tok2id[T_UNK] = self.T_UNK = len(tok2id)
        tok2id[V_UNK] = self.V_UNK = len(tok2id)
        tok2id[C_UNK] = self.C_UNK = len(tok2id)           
        tok2id[V_NULL] = self.V_NULL = len(tok2id)
        tok2id[C_NULL] = self.C_NULL = len(tok2id)           
     
        self.tok2id = tok2id
        self.id2tok = {v: k for k, v in tok2id.items()}
        self.ntokens = len(tok2id)
 
  
    def vectorize(self, examples):
        vec_examples = []
        for ex in examples:
            combined = {'id': [], 'type': [], 'value': [], 'children': []}
            for node in ex:
                if node == 0:
                    continue
                i = I_PREFIX + str(node['id'])
                id_id = [self.tok2id[i] if i in self.tok2id else self.I_UNK]
                t = T_PREFIX + node['type']
                type_id = [self.tok2id[t] if t in self.tok2id else self.T_UNK]
                v = node.get('value', None)
                value_id = [V_NULL]
                if v is not None:
                    try:
                        v = V_PREFIX + str(v)
                    except:
                        v = V_PREFIX + NON_ASCII
                    value_id = [self.tok2id[v] if v in self.tok2id else self.V_UNK]
                c = node.get('children', None)
                children_id = [C_NULL]         
                if c is not None:
                    c = C_PREFIX + str(c)
                    children_id = [self.tok2id[c] if c in self.tok2id else self.C_UNK]
                combined['id'].extend(id_id)
                combined['type'].extend(type_id)
                combined['value'].extend(value_id)
                combined['children'].extend(children_id)
            vec_examples.append(combined)
        return vec_examples


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
    print train_set
    dev_set = code_comp.vectorize(dev_set)
    test_set = code_comp.vectorize(test_set)
    print "took {:.2f} seconds".format(time.time() - start)

    print "Preprocessing training data"
    train_examples = code_comp.create_instances(train_set)
    
    return code_comp, train_examples, dev_set, test_set
