import pickle
from tree_utils import ast_to_lcrs, tree_traversal
from code_comp_utils import vectorize, read_json, create_tok2id

def main():
    train_set = read_json('../data/programs_training.json')
    print "read the training set!"

    tok2id = create_tok2id(train_set)
    print "created the tok2id mapping!"
    with open('../data/tok2id.pickle', 'wb') as f:
        pickle.dump(tok2id, f, protocol=pickle.HIGHEST_PROTOCOL)
    print "dumped the tok2id"

    train_set = vectorize(train_set, tok2id)
    print "vectorized the train set"
    with open('../data/train_vectorized.pickle', 'wb') as f:
        pickle.dump(train_set, f, protocol=pickle.HIGHEST_PROTOCOL)
    print "dumped the train set"

    print "vectorized the dev set"
    dev_set = vectorize(read_json('../data/programs_dev.json'), tok2id)
    with open('../data/dev_vectorized.pickle', 'wb') as f:
        pickle.dump(dev_set, f, protocol=pickle.HIGHEST_PROTOCOL)
    print "dumped the dev set"

    print "vectorized the test set"
    test_set = vectorize(read_json('../data/programs_test.json'), tok2id)
    with open('../data/test_vectorized.pickle', 'wb') as f:
        pickle.dump(test_set, f, protocol=pickle.HIGHEST_PROTOCOL)
    print "dumped the test set"

if __name__ == "__main__":
    main()
