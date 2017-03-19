import os
import re

INDIRECTORY = "results_grid/lstmAsum_fn/"

def main():
	data = []
	for _, dirs, _ in os.walk(INDIRECTORY):
		for d in dirs:
			with open(INDIRECTORY + d + '/real_results.txt', 'r') as f:
				values = []
				for line in f:
					values.extend(re.findall(r'\d+\.*\d*', line))
				values.pop(3) # We don't want that epoch value
				data.append(values)

	with open('grid_lstmAsum_fn_loss.txt', 'w') as f:
		for ex in data:
			f.write(' '.join(ex) + '\n')

if __name__ == "__main__":
	main()
