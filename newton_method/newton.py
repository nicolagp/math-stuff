import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import sys
import os
import subprocess

def rgb(pixels):
    # c1 = [0.125, 0.125, 0.25]
    # c2 = [0.33, 0.22, 0.4]
    # c3 = [1, 0.39, 0.39]
    # c4 = [0, 0.74, 0.41]

    c1 = [0.31, 0.34, 0.34]
    c2 = [0.26, 0.37, 0.37]
    c3 = [0.16, 0.39, 0.445]
    c4 = [0.14, 0.24, 0.61]


    red = np.zeros(pixels.shape)
    green = np.zeros(pixels.shape)
    blue = np.zeros(pixels.shape)

    # set nans to black
    red += np.where(pixels == -1, 0, 0)
    green += np.where(pixels == -1, 0, 0)
    blue += np.where(pixels == -1, 0, 0)

    # # color 1
    # red += c1[0]*np.where(np.logical_and(pixels >= 0 , pixels < 1), (1-pixels), 0)
    # green += c1[1]*np.where(np.logical_and(pixels >= 0 , pixels < 1), (1-pixels), 0)
    # blue += c1[2]*np.where(np.logical_and(pixels >=0 , pixels < 1), (1-pixels), 0)

    # # color 2
    # red += c2[0]*np.where(np.logical_and(pixels >= 1 , pixels < 2), (1-pixels%1), 0)
    # green += c2[1]*np.where(np.logical_and(pixels >= 1 , pixels < 2), (1-pixels%1), 0)
    # blue += c2[2]*np.where(np.logical_and(pixels >= 1 , pixels < 2), (1-pixels%1), 0)

    # # color 3
    # red += c3[0]*np.where(np.logical_and(pixels >= 2 , pixels < 3), (1-pixels%2), 0)
    # green += c3[1]*np.where(np.logical_and(pixels >= 2 , pixels < 3), (1-pixels%2), 0)
    # blue += c3[2]*np.where(np.logical_and(pixels >=2 , pixels < 3), (1-pixels%2), 0)
 
    # # color 4
    # red += c4[0]*np.where(np.logical_and(pixels >= 3 , pixels < 4), (1-pixels%3), 0)
    # green += c4[1]*np.where(np.logical_and(pixels >= 3 , pixels < 4), (1-pixels%3), 0)
    # blue += c4[2]*np.where(np.logical_and(pixels >=3 , pixels < 4), (1-pixels%3), 0)

    # color 1
    red += c1[0]*np.where(np.logical_and(pixels >= 0 , pixels < 1), 1, 0)
    green += c1[1]*np.where(np.logical_and(pixels >= 0 , pixels < 1), 1, 0)
    blue += c1[2]*np.where(np.logical_and(pixels >=0 , pixels < 1), 1, 0)

    # color 2
    red += c2[0]*np.where(np.logical_and(pixels >= 1 , pixels < 2), 1, 0)
    green += c2[1]*np.where(np.logical_and(pixels >= 1 , pixels < 2), 1, 0)
    blue += c2[2]*np.where(np.logical_and(pixels >= 1 , pixels < 2), 1, 0)

    # color 3
    red += c3[0]*np.where(np.logical_and(pixels >= 2 , pixels < 3), 1, 0)
    green += c3[1]*np.where(np.logical_and(pixels >= 2 , pixels < 3), 1, 0)
    blue += c3[2]*np.where(np.logical_and(pixels >=2 , pixels < 3), 1, 0)
 
    # color 4
    red += c4[0]*np.where(np.logical_and(pixels >= 3 , pixels < 4), 1, 0)
    green += c4[1]*np.where(np.logical_and(pixels >= 3 , pixels < 4), 1, 0)
    blue += c4[2]*np.where(np.logical_and(pixels >=3 , pixels < 4), 1, 0)

    # stack and convert to uint8
    stacked = np.dstack((red,green,blue))
    stacked = (stacked * 255.999).astype(np.uint8)
    return stacked

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
    pixels = np.where(np.isnan(pixels), -1, pixels)
    pixels = rgb(pixels)

    plt.imshow(pixels)
    plt.ylabel("$Im(z)$")
    plt.xlabel("$Re(z)$")
    plt.title("$f(z) = z^4 - 1$")
    plt.xticks([])
    plt.yticks([])
    plt.savefig(path)


if __name__ == "__main__":
    main(sys.argv)