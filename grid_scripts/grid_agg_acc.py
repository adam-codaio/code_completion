import os
import re

INDIRECTORY = "results_grid/lstmAsum_fn/"

def main():
        data = []
        for _, dirs, _ in os.walk(INDIRECTORY):
                for d in dirs:
                        entry = []
                        with open(INDIRECTORY + d + '/real_results.txt', 'r') as f:
                                values = []
                                for line in f:
                                        values.extend(re.findall(r'\d+\.*\d*', line))
                                entry.extend(values[:3])

                        with open(INDIRECTORY + d + '/model.weights0.output', 'r') as f:
                                values = []
                                for line in f:
                                        values.extend(re.findall(r'\d+\.*\d*', line))
                                entry.extend(values[-2:])

                        data.append(entry)

        with open('grid_lstmAsum_fn_acc.txt', 'w') as f:
                for ex in data:
                        f.write(' '.join(ex) + '\n')

if __name__ == "__main__":
        main()
