import json
import os
import pickle
import time
from tree_utils import ast_to_lcrs, tree_traversal
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import numpy as np

"""
Largely inspired by code from cs224n PSET 2 for the 
dependency parser
"""

UNK = "<UNK>"

class Config(object):
    data_path = './data'
    segment_size = 50
    train_file = 'programs_training.json'
    dev_file = 'programs_dev.json'
    test_file = 'programs_test.json'
    tok2id_file = 'pickles/tok2id.pickle'
    id2tok_file = 'pickles/id2tok.pickle'
    embeddings_file = 'pickles/wordVectors.pickle'

config = Config()

class CodeCompleter(object):
    def __init__(self):
        with open(os.path.join(config.data_path, config.tok2id_file), 'rb') as f:
            self.tok2id = pickle.load(f)
        with open(os.path.join(config.data_path, config.id2tok_file), 'rb') as f:
            self.id2tok = pickle.load(f)

def vectorize(examples, tok2id):
    vec_examples = []
    num_examples = 0
    for ex in examples:
        num_examples += 1
        if num_examples % 1000 == 0:
            print num_examples
        vec_features = [tok2id.get(feature, tok2id[UNK]) for feature in ex[0]]
        vec_label = [tok2id.get(ex[1][0], tok2id[UNK])]
        vec_examples.append([vec_features, vec_label])
    return vec_examples

def count_line(token_list):
    '''
    Takes the token_list and separates it into features and label for each
    segment.
    '''
    counts = np.zeros(501777)
    token_list = token_list[:-1] # Drop last token which is always Program NT
    for i in xrange(0, len(token_list), config.segment_size):
        segment = token_list[i:i + config.segment_size]
        label = tok2id.get(segment[-1], tok2id[UNK])
        segment = segment[:-1]
        for feature in segment:
            counts[tok2id.get(feature, tok2id[UNK])] += 1
    return counts, label

def get_counts(infile, reduced=False, num_examples=None):
    '''
    This reads in the ast from the file and converts them to binary trees
    and then token lists.
    '''
    counts = []
    labels = []
    num_examples = 0

    with open(infile) as f:
        for line in f:
            num_examples += 1
            if num_examples % 1000 == 0:
                print num_examples
            data = json.loads(line)
            binarized = ast_to_lcrs(data)
            token_list = tree_traversal(binarized)
            seg_count, seg_label = count_line(token_list)
            counts.append(seg_count)
            labels.append(seg_label)
            if reduced:
                num_examples -= 1
                if num_examples == 0:
                    break

    return np.array(counts), np.array(labels)

def score_and_train(counts, labels):
    transformer = TfidfTransformer(smooth_idf=False)
    tfidf = transformer.fit_transform(counts)
    clf = MultinomialNB().fit(tfidf, labels)
    return transformer, clf

def predict(transfomer, classifier, test_input):
    test_tfidf = transformer.transform(test_input)
    predictions = classifier.predict(test_tfidf)
    return predicitons

def eval_preds(predicitons, labels):
    code_comp = CodeCompleter()
    for label, label_ in zip(labels, predicitons):
        wiggle_comp = 0
        if code_comp.id2tok[label][0] == code_comp.id2tok[label_][0]:
                wiggle_comp = 1
        wiggle_preds += wiggle_comp
        correct_preds += (label == label_)
        total_preds += 1
    
    return wiggle_preds / total_preds, correct_preds / total_preds


def main():
   
    counts, labels = get_counts(config.train_file)
    transformer, clf = score_and_train(counts, labels)
    test_counts, test_labels = get_counts(config.test_file)
    preds = predict(transformer, clf, test_file)
    wiggle, accuracy = eval_preds(preds, test_labels)
    print "Wiggle: ", wiggle
    print "Accuracy: ", accuracy

if __name__ == "__main__":
    main()


