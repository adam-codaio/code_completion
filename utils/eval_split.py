# Splits the eval set into a dev (20000) and test (30000) set
# Probably could've done this with some unix utility but yolo
def main():
    with open('../data/programs_eval.txt') as infile:
        with open('../data/programs_test.txt', 'w') as eval_out:
            eval_size = 30000
            for line in infile:
                eval_out.write(line)
                eval_size -= 1
                if eval_size % 1000 == 0:
                    print eval_size, " remaining lines"
                if eval_size == 0:
                    break
        with open('../data/programs_dev.txt', 'w') as dev_out:
            dev_size = 20000
            for line in infile:
                dev_out.write(line)
                dev_size -= 1
                if dev_size % 1000 == 0:
                    print dev_size, " remaining lines"
                if dev_size == 0:
                    break
    with open('../data/programs_eval.json') as infile:
        with open('../data/programs_test.json', 'w') as eval_out:
            eval_size = 30000
            for line in infile:
                eval_out.write(line)
                eval_size -= 1
                if eval_size % 1000 == 0:
                    print eval_size, " remaining lines"
                if eval_size == 0:
                    break
        with open('../data/programs_dev.json', 'w') as dev_out:
            dev_size = 20000
            for line in infile:
                dev_out.write(line)
                dev_size -= 1
                if dev_size % 1000 == 0:
                    print dev_size, " remaining lines"           
                if dev_size == 0:
                    break

if __name__ == '__main__':
    main()
