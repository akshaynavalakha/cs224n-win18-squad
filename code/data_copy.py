import argparse


def copyData(f, lines, new_file):
    start = 0
    end = int(lines)

    count = start
    fileout = open(new_file, "w+")
    for line in f.readlines()[count:(count + end)]:
        fileout.write(str(line))
        #print line
    fileout.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stat-SQuAD ')
    parser.add_argument('data_file', help='data file')
    parser.add_argument('lines', help='lines')
    parser.add_argument('new_data_file', help='new data file')
    args = parser.parse_args()

    with open(args.data_file) as data:
        copyData(data, args.lines, args.new_data_file)