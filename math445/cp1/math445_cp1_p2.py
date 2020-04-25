import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

def main():
    # 1 Plot graphs of S_n(x) for n = 10, 50, 100
    vals = [10, 50, 100]
    for n in vals:
        plot_S(n, "S_{}.png".format(str(n)))


"""
x (float): value to compute fourier Series
n (int): number for approximation
"""
def S_n(x, n):
    result = 0
    for k in range(1, n+1):
        result += compute_bk(k)*np.sin(k*x)

    return result

"""
steps (int): number of steps to approximate integral
k (int): index of coefficient
"""
def compute_bk(k):
    i = integrate.quad(lambda x: x*np.sin(k*x), -np.pi, np.pi)[0]
    return (1/np.pi) * i

"""
n (int): n approximation of fourier series
filename (str): name of file to save plot
"""
def plot_S(n, filename):
    # divide interval:
    x = np.linspace(-np.pi, np.pi, 1000)
    s = np.vectorize(lambda x: S_n(x, n))
    y = s(x)
    p = (x[np.argmax(y)], y.max())

    fig = plt.figure()
    ax = plt.subplot(111)
    ax.plot(x,y)
    ax.annotate('Max: S_{}({:.3f}) = {:.3f}'.format(n, p[0], p[1]), xy = p,
                xycoords='data', xytext=(0.6, 0.9), textcoords='axes fraction',
                arrowprops=dict(facecolor='black', shrink=0.05),
                horizontalalignment='right', verticalalignment='top'
                )
    ax.set_xlabel('x')
    ax.set_ylabel('S_{}(x)'.format(str(n)))
    ax.set_title('S_{}(x) on interval [-pi, pi]'.format(str(n)))
    fig.savefig(filename)

if __name__ == '__main__':
    main()
