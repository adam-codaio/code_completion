import json
import pickle
import numpy as np
from collections import Counter
from tree_utils import ast_to_lcrs, tree_traversal
# from code_comp_utils import vectorize, read_json, create_tok2id

UNK = "<UNK>"
GLOVE_FILE_PATH = '../data/vectors.txt'

def saveWordVectors():
    return loadWordVectors()

def loadWordVectors(filepath=GLOVE_FILE_PATH, dimensions=50):
    """Read pretrained GloVe vectors"""
    wordVectors = {}
    count = 10
    with open(filepath) as ifs:
        for line in ifs:
            line = line.strip()
            if not line:
                continue
            row = line.split()
            token = row[0]
            if token in wordVectors:
            	continue
            data = [float(x) for x in row[1:]]
            if len(data) != dimensions:
                raise RuntimeError("wrong number of dimensions")
            wordVectors[token] = np.asarray(data)
    return wordVectors

def build_embedding_counts():
	allWordVectors = saveWordVectors()
	print "build initial large word vector dictionary..."
	terminal_counts = Counter()
	non_terminal_types = {}
	tok2id = {}
	id2tok = {}

	id2tok[len(tok2id)] = UNK
	tok2id[UNK] = len(tok2id)

	with open('../data/programs_training.json', 'r') as f:
		for line in f:
			data = json.loads(line)
			for node in data:
				if node != 0:
					get_tok2id(node, terminal_counts, non_terminal_types, tok2id, id2tok, allWordVectors)

			# break

	top_terminals = terminal_counts.most_common(50000)

	print "building little dictionaries and word vectors"
        lil_tok2id = {}
        lil_id2tok = {}
        lil_wordVectors = []
        for terminal_id, _ in top_terminals:
            lil_id2tok[len(lil_tok2id)] = id2tok[terminal_id]
            lil_tok2id[id2tok[terminal_id]] = len(lil_tok2id)
            lil_wordVectors.append(allWordVectors[id2tok[terminal_id]])

	val = len(lil_tok2id)
        lil_id2tok[val] = UNK
        lil_tok2id[UNK] = val
	lil_wordVectors[val] = np.asarray(np.random.randn(1, 50), dtype=np.float32)
        print "UNK id is: %d" % lil_tok2id[UNK]

	print "dealing wit non terminals"
	non_terminals = {}
	for data_type in non_terminal_types:
		N_arr = [(data_type, 0, 0), (data_type, 1, 0), (data_type, 0, 1), (data_type, 1, 1)]
		for N_t in N_arr: 
			if N_t not in lil_tok2id:
				val = len(lil_tok2id)
				lil_id2tok[val] = N_t
				lil_tok2id[N_t] = val
				lil_wordVectors[val] = np.asarray(np.random.randn(1, 50), dtype=np.float32)

	print "saving top terminal nodes to file.... format is: id -> count"
	with open('../data/top_terminal_nodes.pickle', 'wb') as counts_pickle:
		pickle.dump(top_terminals, counts_pickle, protocol=pickle.HIGHEST_PROTOCOL)

	print "saving non terminal types to file.... format is: type -> number (num isn't relevant)"
	with open('../data/non_terminal_types.pickle', 'wb') as non_term_pickle:
		pickle.dump(non_terminal_types, non_term_pickle, protocol=pickle.HIGHEST_PROTOCOL)

	print "saving tok2id to file... format is: token -> id (should include all terminal nodes with a embedding and all nonterminal nodes)"
	with open('../data/tok2id.pickle', 'wb') as tok2id_pickle:
		pickle.dump(lil_tok2id, tok2id_pickle, protocol=pickle.HIGHEST_PROTOCOL)

	print "saving id2tok to file... format is: id -> token (should include all terminal nodes with a embedding and all nonterminal nodes)"
	with open('../data/id2tok.pickle', 'wb') as id2tok_pickle:
		pickle.dump(lil_id2tok, id2tok_pickle, protocol=pickle.HIGHEST_PROTOCOL)
	
        print "saving reduced word vectors, total: %d" % len(lil_wordVectors)
        with open('../data/wordVectors.pickle', 'wb') as pkl:
		pickle.dump(lil_wordVectors, pkl, protocol=pickle.HIGHEST_PROTOCOL)

def get_tok2id(data, terminal_counts, non_terminal_types, tok2id, id2tok, allWordVectors):		
	T_i = "EMPTY"
	v = data.get("value", False) 
	if v:
		try:
			v = str(v)
		except:
			v = '<NON_ASCII>'
		T_i = v

	if T_i in allWordVectors:
		if T_i not in tok2id:
			id2tok[len(tok2id)] = T_i
			tok2id[T_i] = len(tok2id)
		terminal_counts[tok2id[T_i]] += 1

	N_type = data["type"]
	if N_type not in non_terminal_types:
		non_terminal_types[N_type] = len(non_terminal_types)
		
	return 

build_embedding_counts()
