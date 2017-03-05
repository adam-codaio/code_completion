import pickle
from collections import Counter
from sets import Set

embed_size = 1500
non_terminals = 200

def build_embedding_counts():
	terminal_counts = Counter()
	non_terminal_types = {}

	with open('../../data/programs_training.json', 'r') as f:
		for line in f:
			import json
			data = json.loads(line)
			for node in data:
				if node != 0:
					non_terminal_count = get_terminal_count(node, terminal_counts, non_terminal_types)

			# break

	top_terminals = terminal_counts.most_common(50000)

	top_terminal_names = {}
	idx_count = 1
	for terminal_node, count in top_terminals:
		top_terminal_names[terminal_node] = idx_count
		idx_count += 1

	print top_terminal_names
	print non_terminal_types 

	with open('terminal_embeddings_idx.pickle', 'wb') as counts_pickle:
		pickle.dump(top_terminals, counts_pickle, protocol=pickle.HIGHEST_PROTOCOL)


	with open('non_terminal_types.pickle', 'wb') as non_term_pickle:
		pickle.dump(non_terminal_types, non_term_pickle, protocol=pickle.HIGHEST_PROTOCOL)

	# print "DONE"
	# print terminal_counts

def get_terminal_count(data, terminal_counts, non_terminal_types):		
	T_i = "EMPTY"
	v = data.get("value", False) 
	if v:
		try:
			v = str(v)
		except:
			v = '<NON_ASCII>'
		T_i = v

	if data["type"] not in non_terminal_types:
		non_terminal_types[data["type"]] = len(non_terminal_types)

	terminal_counts[T_i] += 1

	return 

def embeddings_matrix():
	build_embedding_counts()
	#embeddings[0] = non common terminal nodes
	#embeddings[1-50000] = common terminal nodes - mapping of terminal node to index can be found from top_terminal_names
	#embeddings[50000-50200] = non-terminal nodes
	#NEED TO IMPORT NP
	embeddings = np.array(np.random.randn(50200, 1500), dtype=np.float32)


#data_util.py from assign3 -- for load embeddings
#build 50,000 X J (1500) random numbers from 0-1 matrix for T
#build 97 X J (1500) random numbers from 0-1 matrix for N
#save those to a pickle
#save top 50,000 T to a pickle
embeddings_matrix()