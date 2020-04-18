#include <complex>
#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>
#include "math.h"

#define PI 3.14159265

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
    double epsilon = 1.48e-04;

    for (unsigned int i=0; i<roots.size(); i++) {
        std::complex<double> root = roots[i];
        if (std::abs(z-root) < epsilon) {
            return i;
        }
    }

    // not root
    return -1;
}

std::complex<double> f1(std::complex<double> z) {
    return std::pow(z, 4) - 1.0;
}

std::complex<double> f1_prime(std::complex<double> z) {
    return 4.0 * std::pow(z,3);
}

std::complex<double> f2(std::complex<double> z) {
    return std::sin(z);
}

std::complex<double> f2_prime(std::complex<double> z) {
    return std::cos(z);
}

std::complex<double> f3(std::complex<double> z) {
    return std::tan(z);
}

std::complex<double> f3_prime(std::complex<double> z) {
    return 1.0 / std::pow(std::cos(z), 2);
}

std::complex<double> newton(std::complex<double> z0) {
    double epsilon = 1.48e-06;
    std::complex<double> z_new = z0;
    int iters = 0;
    int maxiter = 50;
    // bool success = false;

    for (int i=0; i<maxiter; i++) {
        // check f_prime not zero
        // if (std::abs(f_prime(z_new)) == 0) {
        //     return std::complex<double>(std::nan(""), std::nan(""));
        // }

        // update z
        z_new = z_new - (f3(z_new) / f3_prime(z_new));
        iters++;
    
        // test for convergence
        if (std::abs(z_new) < epsilon) {
            // success = true;
            break;
        }
    }

    // if (!success) {
    //    return std::complex<double>(std::nan(""), std::nan(""));
    // }

    return z_new;
}



int main(int argc, char* argv[]) {
    /* Arguments:
    int n: dimension of points to calculate
    string filepath: where to save image
    */

    // parse arguments
    int n;
    char* path;

    if (argc != 3) {
        std::cerr << "Wrong number of arguments." << std::endl;
        std::cerr << "Usage: ./newton n filepath" << std::endl;
    }
    else {
        n = std::atoi(argv[1]);
        path = argv[2];
        std::cout << "Arguments: " << n << ", " << path << std::endl;
        return 0;
    }

    // number of points
    std::vector<double> re = linspace((3.14/2)-0.2, (3.14/2)+0.2, n);
    std::vector<double> im = linspace(-0.2, 0.2, n);

    // roots
    std::vector<std::complex<double>> roots1;
    roots1.push_back(std::complex<double>(1, 0));
    roots1.push_back(std::complex<double>(-1, 0));
    roots1.push_back(std::complex<double>(0, 1));
    roots1.push_back(std::complex<double>(0, -1));

    std::vector<std::complex<double>> roots2;
    roots2.push_back(std::complex<double>(0, 0));
    roots2.push_back(std::complex<double>(PI, 0));


    // save img
    std::ofstream out("out.csv");
    for (int i=0; i<n; i++) {
        for (int j=0; j<n; j++) {
            std::complex<double> z = newton(
                std::complex<double>(re[j], im[i]));

            out << find_roots(roots2, z) << ", ";
        }
        out << "\n";
    }

    return 0;
}
