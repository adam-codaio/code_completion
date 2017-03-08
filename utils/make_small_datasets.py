TRAIN_SIZE = 500
TEST_SIZE = 100

def write_small(outfile, infile, size):
    with open(outfile, 'w') as of:
        with open(infile, 'r') as inf:
            examples = TRAIN_SIZE
            for line in inf:
                of.write(line)
                examples -= 1
                if examples == 0:
                    break

write_small('../train_nt_vectorized_small.txt', '../data/train_nt_vectorized.txt', TRAIN_SIZE)
write_small('../train_t_vectorized_small.txt', '../data/train_t_vectorized.txt', TRAIN_SIZE)
write_small('../test_nt_vectorized_small.txt', '../data/test_nt_vectorized.txt', TEST_SIZE)
write_small('../test_t_vectorized_small.txt', '../data/test_t_vectorized.txt', TEST_SIZE)
