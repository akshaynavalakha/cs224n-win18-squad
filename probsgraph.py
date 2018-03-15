import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
import argparse
import time

#https://matplotlib.org/examples/color/colormaps_reference.html

def getProbDisplay(contextlist, probabilitylist, file):
    contextlist=np.array(contextlist)
    probabilitylist=np.array(probabilitylist)
    return getProbDisplay1(contextlist, probabilitylist)

def getProbDisplay1(contextlist, probabilitylist, file_name="test"):
    y_pos = np.arange(len(contextlist))
    colours = cm.Greens(probabilitylist / max(probabilitylist))
    p = plt.scatter(y_pos, probabilitylist, alpha=0.5, c=probability, cmap='Greens')
    plt.clf()
    plt.colorbar(p)
    plt.bar(range(len(probabilitylist)), [1]*len(contextlist), color = colours)

    plt.xticks(y_pos, contextlist)
    #plt.ylabel('probability')
    #plt.title('Category probabilities')
    plt.savefig(file_name + str(int(time.time())))   # save the figure to file



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='probability graph for SQuAD ')
    parser.add_argument('final_file', default="test",const="test", nargs='?',help='Image file , ex abc.png')
    args = parser.parse_args()
    sample=['Where','is','Paris','city','in','here']
    probability=[0.1, 0.05, 0.05, 0.2, 0.4, 0.2]
    getProbDisplay(sample, probability, args.final_file)
