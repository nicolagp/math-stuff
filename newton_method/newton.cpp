#include <complex>
#include <iostream>
#include <chrono>
#include "math.h"

std::complex<double> f(std::complex<double>);
std::complex<double> f_prime(std::complex<double>);
std::complex<double> newton(std::complex<double>);

std::complex<double> f(std::complex<double> z) {
    return std::pow(z, 3) + z;
}

std::complex<double> f_prime(std::complex<double> z) {
    return 3.0 * std::pow(z,2) + 1.0;
}

std::complex<double> newton(const std::complex<double> z0) {
    double epsilon = 1.48e-08;
    std::complex<double> z_new = z0;
    int iters = 0;
    int maxiter = 50;
    bool success = false;

    for (int i=0; i<maxiter; i++) {
        // update z
        z_new = z_new - (f(z_new) / f_prime(z_new));
        iters++;

        // test for convergence
        if (std::abs(z_new) < epsilon) {
            success = true;
            break;
        }
    }

    if (success) {
        // std::cout << "Converged to " << z_new << "in " << iters << " iterations" << std::endl;
    }
    else {
        // std::cout << "Failed to converge in " << maxiter << "iterations" << std::endl;
    }

    return z_new;

}

int main() {
    std::complex<float> z0(10, -2); 
    auto t1 = std::chrono::high_resolution_clock::now();
    for (int i=0; i < 10000; i++) {
        newton(z0);
        z0 += 1.0;
    }
    newton(z0);
    
    auto t2 = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();

    std::cout << "Finished in: " << duration << " microseconds" << std::endl;
    return 0;
}
