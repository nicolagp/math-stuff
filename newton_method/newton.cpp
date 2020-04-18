#include <complex>
#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>
#include "math.h"

using namespace std;

#define PI 3.14159265

complex<double> f(complex<double>);
complex<double> f_prime(complex<double>);
pair<complex<double>, int> newton(complex<double>);
vector<double> linspace(double, double, double);
int find_roots(vector<complex<double>>, complex<double>);
double pixel_value(vector<complex<double>>, pair<complex<double>, int>);

vector<double> linspace(double start, double end, double len) {
    vector<double> result;
    double delta = (end-start)/double(len-1);

    for(int i=0; i<len; i++) {
        result.push_back(start + i*delta);
    }

    return result;
}

/*
Returns the position of the given root in the list of roots. If it can't find
the root, it returns -1.

roots: list with the roots of the function.
z: number to find position in list.
*/
int find_roots(vector<complex<double>> roots,
complex<double> z) {
    double epsilon = 1.48e-04;

    for (unsigned int i=0; i<roots.size(); i++) {
        complex<double> root = roots[i];
        if (abs(z-root) < epsilon) {
            return i;
        }
    }

    // not root
    return -1;
}

complex<double> f(complex<double> z) {
    return pow(z, 4) - 1.0;
}

complex<double> f_prime(complex<double> z) {
    return 4.0 * pow(z,3);
}

/*
Generates pixel value from result of newton's method calculation.

roots: list with the roots of the function.
p: pait of the complex root, and iterations taken.
*/
double pixel_value(vector<complex<double>> roots,
    pair<complex<double>, int> p) {

    // check convergence
    if (p.first == complex<double>(nan(""), nan(""))) {
        return nan("");
    }

    // return position in list + shading if found root
    double pos = (double) find_roots(roots, p.first);
    if (pos == -1) {
        return nan("");
    }

    return pos + ((double) p.second / 50);
}

/*
Compute newton method of f starting at z0.
Arguments:
- complex<double> z0: complex number to start newton method from

Returns:
- Pair of the complex root achieved and the number of iterations until
convergence. In the case it doesn't converge or reaches a derivative which
is equal to 0, it returns a pair (nan,0). 
*/
pair<complex<double>, int> newton(complex<double> z0) {
    double epsilon = 1.48e-08;
    complex<double> z_new = z0;
    pair<complex<double>, int> result;
    int iters = 0;
    int maxiter = 50;
    bool converged = false;

    for (int i=0; i<maxiter; i++) {
        // check f_prime not zero
        if (abs(f_prime(z_new)) == 0) {
            break;
        }

        // update z and iteration
        z_new = z_new - (f(z_new) / f_prime(z_new));
        iters++;
    
        // test for convergence
        if (abs(f(z_new)) < epsilon) {
            converged = true;
            result.first = z_new;
            result.second = iters;
            break;
        }
    }

    if (!converged) {
        result.first = complex<double>(nan(""), nan(""));
        result.second = 0;
    }

    return result;
}


/* 
Arguments:
int n: dimension of points to calculate
*/
int main(int argc, char* argv[]) {
    // parse arguments
    int n;

    if (argc != 2) {
        cerr << "Wrong number of arguments." << endl;
        cerr << "Usage: ./newton n" << endl;
        return 1;
    }
    else {
        n = atoi(argv[1]);
    }

    // number of points
    vector<double> re = linspace(-2, 2, n);
    vector<double> im = linspace(-2, 2, n);

    // roots
    vector<complex<double>> roots;
    roots.push_back(complex<double>(1, 0));
    roots.push_back(complex<double>(-1, 0));
    roots.push_back(complex<double>(0, 1));
    roots.push_back(complex<double>(0, -1));

    // save img
    ofstream out("out.csv");
    for (int i=0; i<n; i++) {
        for (int j=0; j<n; j++) {
            pair<complex<double>, int> p = newton(
                complex<double>(re[j], im[i]));

            out << pixel_value(roots, p) << ",";
        }
        out << "\n";
    }

    return 0;
}
