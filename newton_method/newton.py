import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import subprocess

def main(argv):
    if len(argv) != 3:
        print("Wrong number of arguments.")
        print("Usage: python newton.py n imgpath")
        return 1
    else:
        n = argv[1]
        path = argv[2]

    
    # call c++
    subprocess.call("./newton.o {}".format(n), shell=True)

    # load pixel values
    pixels = pd.read_csv('out.csv', sep=',',header=None)
    pixels = pixels.drop(int(n), axis=1).values

    # plotting
    plt.set_cmap('winter')
    current_cmap = plt.cm.get_cmap()
    current_cmap.set_bad(color='black')

    plt.imshow(pixels)
    plt.ylabel("$Im(z)$")
    plt.xlabel("$Re(z)$")
    plt.title("$f(z) = z^4 - 1$")
    plt.xticks([])
    plt.yticks([])
    plt.savefig(path)


if __name__ == "__main__":
    main(sys.argv)