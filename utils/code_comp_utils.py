import json
import os
import time
from tree_utils import ast_to_lcrs, tree_traversal

"""
Largely inspired by code from cs224n PSETs 2 and 3 
"""

UNK = "<UNK>"

class Config(object):
    data_path = './data'
    reduced_train = 500
    reduced_dev = 100
    reduced_test = 100
    segment_size = 50
    nt_pred = True
    train_file = 'programs_training.json'
    dev_file = 'programs_dev.json'
    test_file = 'programs_test.json'

config = Config()

class CodeCompleter(object):
    def __init__(self, dataset):
        tok2id = {};
        for ex in dataset:
            features = ex[0]
            label = ex[1]
            for feature in features:
                if feature not in tok2id:
                    tok2id[feature] = len(tok2id)
        if label[0] not in tok2id:
            tok2id[label[0]] = len(tok2id)
        tok2id[UNK] = len(tok2id)

        self.tok2id = tok2id
  
    def vectorize(self, examples):
        vec_examples = []
        for ex in examples:
            vec_features = [self.tok2id.get(feature, self.tok2id[UNK]) for feature in ex[0]]
            vec_label = [self.tok2id.get(ex[1][0], self.tok2id[UNK])]
            vec_examples.append([vec_features, vec_label])
        return vec_examples

    def create_instances(self, examples):
        return examples

    def extract_features(self, example):
        # There is where a lot of heavy lifting is going to happen
        # called from create_instances
        pass

    def save_pickle(self, dataset):
        with open('tok2id.pickle', 'wb') as tok2id_pickle:
            pickle.dump(self.tok2id, tok2id_pickle, protocol=pickle.HIGHEST_PROTOCOL)

def process_token_list(token_list):
    '''
    Takes the token_list and separates it into features and label for each
    segment.
    '''
    token_list = token_list[:-1] # Drop last token which is always Program NT
    segments = []
    for i in xrange(0, len(token_list), config.segment_size):
        segment = token_list[i:i + config.segment_size]
        features = []
        for tup in segment[:-1]:
            features.extend(list(tup))
        idx = 0 if config.nt_pred else 1
        label = [token_list[-1][idx]]
        segments.append([features, label])
    return segments

def read_json(infile, reduced=False, num_examples=None):
    '''
    This reads in the asts from the file and converts them to binary trees
    and then token lists.
    '''
    examples = []
 
    with open(infile) as f:
        for line in f:
            data = json.loads(line)
            binarized = ast_to_lcrs(data)
            token_list = tree_traversal(binarized)
            segments = process_token_list(token_list)
            examples.extend(segments)
            if reduced:
                num_examples -= 1
                if num_examples == 0:
                    break

    return examples

def load_and_preprocess_data(reduced=True):
    
    print "Loading data...",
    start = time.time()

    train_set = read_json(os.path.join(config.data_path, config.train_file),
                          reduced, config.reduced_train)
    dev_set = read_json(os.path.join(config.data_path, config.dev_file),
                        reduced, config.reduced_dev)
    test_set = read_json(os.path.join(config.data_path, config.test_file),
                         reduced, config.reduced_test)
    print "took {:.2f} seconds".format(time.time() - start)
    
    print "Building Code Completer...",
    start = time.time()
    code_comp = CodeCompleter(train_set)
    print "took {:.2f} seconds".format(time.time() - start)

    print "saving token to id matrix"
    code_comp.save_pickle(train_set)

    print "Vectorizing data...",
    start = time.time()
    train_set = code_comp.vectorize(train_set)
    dev_set = code_comp.vectorize(dev_set)
    test_set = code_comp.vectorize(test_set)
    print "took {:.2f} seconds".format(time.time() - start)

    print "Preprocessing training data"
    train_examples = code_comp.create_instances(train_set)
    
    return code_comp, train_examples, dev_set, test_set
