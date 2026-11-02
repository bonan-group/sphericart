/*
    Tests for the asynchronous CUDA API of Sphericart:
    `compute_with_gradients_async` must only enqueue the kernel without
    synchronizing the stream, `release_device_resources` must free the lazily
    allocated device memory (and be idempotent), and a calculator must reject
    inputs that live on a different CUDA device than the one it is bound to.

    This test only uses the dynamically loaded CUDA runtime through gpu-lite,
    so building it does not require the CUDA toolkit.
*/
#include <algorithm>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <gpulite/gpulite.hpp>

#include "sphericart_cuda.hpp"

int main() {
    int device_count = 0;
    try {
        GPULITE_CUDART_CALL(cudaGetDeviceCount(&device_count));
    } catch (const std::exception& error) {
        std::cout << "No usable CUDA device; skipping async CUDA API test (" << error.what() << ")"
                  << std::endl;
        return 0;
    }
    if (device_count == 0) {
        std::cout << "No CUDA device; skipping async CUDA API test" << std::endl;
        return 0;
    }

    constexpr size_t l_max = 4;
    constexpr size_t n_samples = 200000;
    constexpr size_t n_spherical = (l_max + 1) * (l_max + 1);
    constexpr int n_launches = 64;
    const size_t xyz_size = 3 * n_samples;
    const size_t sph_size = n_samples * n_spherical;
    const size_t dsph_size = 3 * sph_size;

    std::vector<double> xyz(xyz_size);
    for (size_t sample = 0; sample < n_samples; ++sample) {
        xyz[3 * sample] = 0.25 + 0.000001 * static_cast<double>(sample);
        xyz[3 * sample + 1] = -0.5 + 0.000002 * static_cast<double>(sample);
        xyz[3 * sample + 2] = 0.75 - 0.0000015 * static_cast<double>(sample);
    }
    std::vector<double> reference_sph(sph_size);
    std::vector<double> reference_dsph(dsph_size);
    std::vector<double> async_sph(sph_size);
    std::vector<double> async_dsph(dsph_size);

    double* device_xyz = nullptr;
    double* device_sph = nullptr;
    double* device_dsph = nullptr;
    cudaStream_t stream = nullptr;

    try {
        GPULITE_CUDART_CALL(cudaSetDevice(0));
        GPULITE_CUDART_CALL(cudaStreamCreate(&stream));
        GPULITE_CUDART_CALL(
            cudaMalloc(reinterpret_cast<void**>(&device_xyz), xyz_size * sizeof(double))
        );
        GPULITE_CUDART_CALL(
            cudaMalloc(reinterpret_cast<void**>(&device_sph), sph_size * sizeof(double))
        );
        GPULITE_CUDART_CALL(
            cudaMalloc(reinterpret_cast<void**>(&device_dsph), dsph_size * sizeof(double))
        );
        GPULITE_CUDART_CALL(
            cudaMemcpy(device_xyz, xyz.data(), xyz_size * sizeof(double), cudaMemcpyHostToDevice)
        );

        using Calculator = sphericart::cuda::SphericalHarmonics<double>;
        Calculator calculator(l_max);

        // synchronous reference run; this also warms up NVRTC compilation and
        // the lazy allocation of the prefactors on the device
        calculator.compute_with_gradients(device_xyz, n_samples, device_sph, device_dsph, stream);
        GPULITE_CUDART_CALL(cudaMemcpy(
            reference_sph.data(), device_sph, sph_size * sizeof(double), cudaMemcpyDeviceToHost
        ));
        GPULITE_CUDART_CALL(cudaMemcpy(
            reference_dsph.data(), device_dsph, dsph_size * sizeof(double), cudaMemcpyDeviceToHost
        ));

        // enqueue a burst of asynchronous calls and check that they return
        // before the GPU has time to process the whole queue
        const auto enqueue_start = std::chrono::steady_clock::now();
        for (int i = 0; i < n_launches; ++i) {
            calculator.compute_with_gradients_async(
                device_xyz, n_samples, device_sph, device_dsph, stream
            );
        }
        const auto enqueue_elapsed = std::chrono::steady_clock::now() - enqueue_start;

        const auto sync_start = std::chrono::steady_clock::now();
        GPULITE_CUDART_CALL(cudaStreamSynchronize(stream));
        const auto sync_elapsed = std::chrono::steady_clock::now() - sync_start;

        std::cout << "enqueued " << n_launches << " kernels in "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(enqueue_elapsed).count()
                  << " ms; draining the stream took "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(sync_elapsed).count()
                  << " ms" << std::endl;
        if (enqueue_elapsed >= sync_elapsed / 4) {
            std::cerr << "Asynchronous CUDA calls seem to wait for the stream" << std::endl;
            return 1;
        }

        GPULITE_CUDART_CALL(cudaMemcpy(
            async_sph.data(), device_sph, sph_size * sizeof(double), cudaMemcpyDeviceToHost
        ));
        GPULITE_CUDART_CALL(cudaMemcpy(
            async_dsph.data(), device_dsph, dsph_size * sizeof(double), cudaMemcpyDeviceToHost
        ));
        if (async_sph != reference_sph || async_dsph != reference_dsph) {
            std::cerr << "Asynchronous and synchronous CUDA results differ" << std::endl;
            return 1;
        }

        // releasing device resources must be idempotent, and the calculator
        // must transparently re-allocate them on the bound device
        calculator.release_device_resources();
        calculator.release_device_resources();
        calculator.compute_with_gradients_async(
            device_xyz, n_samples, device_sph, device_dsph, stream
        );
        GPULITE_CUDART_CALL(cudaStreamSynchronize(stream));
        GPULITE_CUDART_CALL(cudaMemcpy(
            async_sph.data(), device_sph, sph_size * sizeof(double), cudaMemcpyDeviceToHost
        ));
        GPULITE_CUDART_CALL(cudaMemcpy(
            async_dsph.data(), device_dsph, dsph_size * sizeof(double), cudaMemcpyDeviceToHost
        ));
        if (async_sph != reference_sph || async_dsph != reference_dsph) {
            std::cerr << "CUDA results changed after releasing device resources" << std::endl;
            return 1;
        }

        if (device_count > 1) {
            double* other_device_xyz = nullptr;
            GPULITE_CUDART_CALL(cudaSetDevice(1));
            GPULITE_CUDART_CALL(
                cudaMalloc(reinterpret_cast<void**>(&other_device_xyz), xyz_size * sizeof(double))
            );

            bool migration_rejected = false;
            try {
                calculator.compute_with_gradients_async(
                    other_device_xyz, n_samples, device_sph, device_dsph, stream
                );
            } catch (const std::runtime_error& error) {
                migration_rejected =
                    std::string(error.what()).find("migration is unsupported") != std::string::npos;
            }
            if (!migration_rejected) {
                std::cerr << "Calculator accepted migration to another CUDA device" << std::endl;
                return 1;
            }

            GPULITE_CUDART_CALL(cudaFree(other_device_xyz));
            GPULITE_CUDART_CALL(cudaSetDevice(0));
        }

        GPULITE_CUDART_CALL(cudaFree(device_dsph));
        GPULITE_CUDART_CALL(cudaFree(device_sph));
        GPULITE_CUDART_CALL(cudaFree(device_xyz));
        GPULITE_CUDART_CALL(cudaStreamDestroy(stream));
    } catch (const std::exception& error) {
        std::cerr << "Unexpected error in async CUDA API test: " << error.what() << std::endl;
        return 1;
    }

    return 0;
}
