import json
import os
import pickle
import time
from tree_utils import ast_to_lcrs, tree_traversal
import numpy as np

"""
Largely inspired by code from cs224n PSET 2 for the 
dependency parser
"""

UNK = "<UNK>"

class Config(object):
    data_path = './data'
    segment_size = 50
    test_file = 'programs_test.json'
    tok2id_file = 'pickles/tok2id.pickle'
    id2tok_file = 'pickles/id2tok.pickle'
    embeddings_file = 'pickles/wordVectors.pickle'

def process_token_list(token_list, NT):
    '''
    Takes the token_list and separates it into features and label for each
    segment.
    '''
    token_list = token_list[:-1] # Drop last token which is always Program NT
    segments = []
    for i in xrange(0, len(token_list), config.segment_size):
        if len(token_list) > 500: continue
        segment = token_list[i:i + config.segment_size]
        features = []
        for tup in segment[:-1]:
            features.extend(list(tup))
        idx = 0 if NT else 1
        label = [segment[-1][idx]]
        segments.append(([features, label], i))
    return np.random.choice(segments, 1)

def read_json(infile, reduced=False, num_examples=None):
    '''
    This reads in the ast from the file and converts them to binary trees
    and then token lists.
    '''
    examples_nt = []
    examples_t = []
    full_ast = []
    num_examples = 0
    rand_samples_count = 0
    with open(infile) as f:
        for line in f:
            num_examples += 1
            if num_examples % 1000 == 0:
                print num_examples
            if rand_samples_count >= 10:
                break
            if np.random.rand(0,1) < .5:
                data = json.loads(line)
                binarized = ast_to_lcrs(data)
                token_list = tree_traversal(binarized)
                segments_nt = process_token_list(token_list, True)
                segments_t = process_token_list(token_list, False)
                examples_nt.extend(segments_nt)
                examples_t.extend(segments_t)
                full_ast.append(line)
                if reduced:
                    num_examples -= 1
                    if num_examples == 0:
                        break

    return examples_nt, examples_t, full_ast

def vectorize_set(dataset, tok2id, path):
    print "about to vectorize the %s set" % path
    vectorized_dataset = vectorize(dataset, tok2id)
    print "vectorized the %s set" % path
    with open('../data/' + path + '_vectorized.txt', 'w') as f:
        for data in vectorized_dataset:
            f.write(repr(data) + '\n')
    print "wrote the %s set" % path

def test_set(tok2id):
    print "reading the test set"
    test_set_nt, test_set_t, ast = read_json('../data/programs_test.json')
    print "read the test set"

    vectorize_set(test_set_nt, tok2id, 'eval_example_nt')
    vectorize_set(test_set_t, tok2id, 'eval_example_t')

    print "saving asts"
    with open('../data/eval_example_ast.txt', 'w') as f:
        json.dump(ast, f)
    print "wrote ast to the ../data/eval_example_ast.txt set"

def main():
    print "fetching the tok2id"
    with open('../data/pickles/tok2id.pickle', 'rb') as f:
        tok2id = pickle.load(f)
    print "read the tok2id mapping"
   
    test_set(tok2id)

if __name__ == "__main__":
    main()

