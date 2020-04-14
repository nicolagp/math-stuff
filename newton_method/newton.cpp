#include <complex>
#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>
#include "math.h"

std::complex<double> f(std::complex<double>);
std::complex<double> f_prime(std::complex<double>);
std::complex<double> newton(std::complex<double>);
std::vector<double> linspace(double, double, double);
int find_roots(std::vector<std::complex<double>>, std::complex<double>);

std::vector<double> linspace(double start, double end, double len) {
    std::vector<double> result;
    double delta = (end-start)/double(len-1);

    for(int i=0; i<len; i++) {
        result.push_back(start + i*delta);
    }

    return result;
}

int find_roots(std::vector<std::complex<double>> roots,
std::complex<double> z) {
    double epsilon = 1.48e-08;

    for (unsigned int i=0; i<roots.size(); i++) {
        std::complex<double> root = roots[i];
        if (std::abs(z-root) < epsilon) {
            return i;
        }
    }

    // not root
    return -1;
}

std::complex<double> f(std::complex<double> z) {
    return std::pow(z, 4) - 1.0;
}

std::complex<double> f_prime(std::complex<double> z) {
    return 4.0 * std::pow(z,3);
}

std::complex<double> newton(std::complex<double> z0) {
    double epsilon = 1.48e-08;
    std::complex<double> z_new = z0;
    int iters = 0;
    int maxiter = 50;
    bool success = false;

    for (int i=0; i<maxiter; i++) {
        // check f_prime not zero
        // if (std::abs(f_prime(z_new)) == 0) {
        //     throw "Division by zero";
        // }

        // update z
        z_new = z_new - (f(z_new) / f_prime(z_new));
        iters++;
    
        // test for convergence
        if (std::abs(z_new) < epsilon) {
            success = true;
            break;
        }
    }

    // if (!success) {
    //    throw "Didn't converge";
    // }

    return z_new;
}



int main() {
    // number of points
    int n = 1000;
    std::vector<double> lin = linspace(-2, 2 , n);
    int img[n][n];

    // roots
    std::vector<std::complex<double>> roots;
    roots.push_back(std::complex<double>(1, 0));
    roots.push_back(std::complex<double>(-1, 0));
    roots.push_back(std::complex<double>(0, 1));
    roots.push_back(std::complex<double>(0, -1));

    // fill img
    for (int i=0; i<n; i++) {
        for (int j=0; j<n; j++) {
            std::complex<double> z = newton(
                std::complex<double>(lin[i], lin[j]));
            img[i][j] = find_roots(roots, z);
        }
    }

    // save img
    std::ofstream out("test.csv");
    for (int i=0; i<n; i++) {
        for (int j=0; j<n; j++) {
            out << img[i][j] << ", ";
        }
        out << "\n";
    }

    return 0;
}
