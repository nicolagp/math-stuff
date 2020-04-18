import numpy as np
import datetime
import matplotlib.pyplot as plt
import sys
import os

def main(argv):
    if len(argv) != 4:
        print("Wrong number of arguments.")
        print("Usage: python newton.py n dpi imgpath")
    else:
        n = argv[1]
        dpi = argv[2]
        path = argv[3]

    
    mat = pd.read_csv('out.csv', sep=',',header=None)
    mat = mat.drop(n, axis=1).values
    plt.matshow(mat, cmap="winter")
    plt.ylabel("$Im(z)$")
    plt.xlabel("$Re(z)$")
    plt.title("$f(z) = tan(z)$")
    plt.xticks([])
    plt.yticks([])
    plt.savefig("tan.jpg", dpi=6000)


if __name__ == "__main__":
    main(sys.argv)