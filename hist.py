from __future__ import print_function
import argparse
import matplotlib.pyplot as plt
import time

def getInfo(file):
    lines=0
    totalwordCount=0
    maxWordsInLine = 0
    wordCountList=[]

    for lineOfText in file:
        wordCount=0
        lines += 1
        lineList = lineOfText.split()
        wordCount =+ len(lineList)
        wordCountList.append(wordCount)
        totalwordCount += wordCount
        if len(lineList) > maxWordsInLine:
            maxWordsInLine = len(lineList)

    print("Average words per line: {}".format(totalwordCount / lines))
    print("Most words in a single line: {}".format(maxWordsInLine))
    c = sorted(wordCountList)
    plt.hist(c, bins=30)
    plt.ylabel('frequency')
    plt.xlabel('Question length')
    plt.grid(True)
    plt.savefig("test" + str(int(time.time())))   # save the figure to file
    #plt.show()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stat-SQuAD ')
    parser.add_argument('data_file', help='data file')
    #parser.add_argument('copy', help='data file')
    args = parser.parse_args()

    with open(args.data_file) as data:
        getInfo(data)


