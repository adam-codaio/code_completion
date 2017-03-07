import pickle
from tree_utils import ast_to_lcrs, tree_traversal
from code_comp_utils import vectorize, read_json

def vectorize_set(dataset, tok2id, path):
    print "about to vectorize the %s set" % path
    vectorized_dataset = vectorize(dataset, tok2id)
    print "vectorized the %s set" % path
    with open('../data/' + path + '_vectorized.txt', 'w') as f:
        for data in vectorized_dataset:
            f.write(repr(data) + '\n')
    print "wrote the %s set" % path

def train_set(tok2id):
    print "reading the training set"
    train_set_nt, train_set_t = read_json('../data/programs_training.json')
    print "read the training set"

    vectorize_set(train_set_nt, tok2id, 'train_nt')
    vectorize_set(train_set_t, tok2id, 'train_t')

def dev_set(tok2id):
    print "reading the dev set"
    dev_set_nt, dev_set_t = read_json('../data/programs_dev.json')
    print "read the dev set"

    vectorize_set(dev_set_nt, tok2id, 'dev_nt')
    vectorize_set(dev_set_t, tok2id, 'dev_t')

def test_set(tok2id):
    print "reading the test set"
    test_set_nt, test_set_t = read_json('../data/programs_test.json')
    print "read the test set"

    vectorize_set(test_set, tok2id, 'test_nt')
    vectorize_set(test_set, tok2id, 'test_t')

def main():
    print "fetching the tok2id"
    with open('../data/pickles/tok2id.pickle', 'rb') as f:
        tok2id = pickle.load(f)
    print "read the tok2id mapping"

    train_set(tok2id)
    dev_set(tok2id)
    test_set(tok2id)

if __name__ == "__main__":
    main()
