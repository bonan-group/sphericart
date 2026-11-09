#include <iostream>
#include <stdexcept>
#include <type_traits>

#include "sphericart_cuda.hpp"

int main() {
    using Calculator = sphericart::cuda::SphericalHarmonics<double>;
    static_assert(
        std::is_same_v<decltype(&Calculator::release_device_resources), void (Calculator::*)()>
    );
    static_assert(std::is_same_v<
                  decltype(&Calculator::compute_with_gradients_async),
                  void (Calculator::*)(const double*, size_t, double*, double*, void*)>);

    try {
        Calculator calculator(1);
    } catch (const std::runtime_error&) {
        return 0;
    }
    std::cerr << "CUDA stub constructor unexpectedly succeeded" << std::endl;
    return 1;
}
