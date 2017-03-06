import os
import re

ROOT_DIR = "../data/data/"

with open('embeddings_corpus.txt', 'w') as out_file:
	for subdir, dirs, files in os.walk(ROOT_DIR):
		for file in files:
			if file.endswith('.js'): 
				file_path = os.path.join(subdir, file)
				if os.path.exists(file_path):
					with open(file_path) as curr_file:
						for line in curr_file:
							#replace all the "code" characters
							line = re.sub('[{};().]', ' ', line)
							line = line.split()
							for word in line:
								out_file.write(word + " ")
