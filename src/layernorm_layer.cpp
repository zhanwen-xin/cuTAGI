#include "../include/layernorm_layer.h"

#include "../include/custom_logger.h"
#include "../include/param_init.h"

#ifdef USE_CUDA
#include "../include/layernorm_layer_cuda.cuh"
#endif
#include <cmath>
#include <thread>

////////////////////////////////////////////////////////////////////////////////
/// CPU kernels for Layer Norm
////////////////////////////////////////////////////////////////////////////////
void layernorm_stat_mean_var(const std::vector<float> &mu_a,
                             const std::vector<float> &var_a, int ni,
                             int start_chunk, int end_chunk,
                             std::vector<float> &mu_s,
                             std::vector<float> &var_s)
/*
 */
{
    // ni in the case of conv2d will be wihi * fi
    for (int col = start_chunk; col < end_chunk; col++) {
        float sum_mu = 0.0f;
        float sum_var = 0.0f;
        for (int i = 0; i < ni; i++) {
            sum_mu += mu_a[col * ni + i];
            sum_var += var_a[col * ni + i];
        }
        mu_s[col] = sum_mu / ni;
        var_s[col] = sum_var;
    }
}

void layernorm_sample_var(const std::vector<float> &mu_a,
                          const std::vector<float> &mu_s,
                          const std::vector<float> &var_s, int ni,
                          int start_chunk, int end_chunk,
                          std::vector<float> &var_sample)
/*
 */
{
    // ni in the case of conv2d will be wihi * fi
    float mu_norm_sum = 0.0f;
    float var_norm_sum = 0.0f;
    for (int col = start_chunk; col < end_chunk; col++) {
        float sum = 0.0f;
        for (int i = 0; i < ni; i++) {
            sum += (mu_a[col * ni + i] - mu_s[col]) *
                   (mu_a[col * ni + i] - mu_s[col]);
        }
        var_sample[col] = (sum + var_s[col]) / (ni - 1);
    }
}

void layernorm_fwd_mean_var(
    const std::vector<float> &mu_w, const std::vector<float> &var_w,
    const std::vector<float> &mu_b, const std::vector<float> &var_b,
    const std::vector<float> &mu_a, const std::vector<float> &var_a,
    const std::vector<float> &mu_ra, const std::vector<float> &var_ra,
    bool bias, float epsilon, int ni, int start_chunk, int end_chunk,
    std::vector<float> &mu_z, std::vector<float> &var_z)
/*
 */
{
    for (int row = start_chunk; row < end_chunk; row++) {
        float inv_sqrt_var_ra = 1.0f / std::sqrt(var_ra[row] + epsilon);
        float mu_ra_term = mu_ra[row];
        for (int col = 0; col < ni; col++) {
            int index = col + row * ni;
            float adjusted_mu_a = mu_a[index] - mu_ra_term;
            float mu_term = adjusted_mu_a * mu_w[col];

            mu_z[index] = inv_sqrt_var_ra * mu_term;
            var_z[index] =
                inv_sqrt_var_ra * inv_sqrt_var_ra *
                (var_a[index] * (mu_w[col] * mu_w[col] + var_w[col]) +
                 var_w[col] * adjusted_mu_a * adjusted_mu_a);
            if (bias) {
                mu_z[index] += mu_b[col];
                var_z[index] += var_b[col];
            }
        }
    }
}

void layernorm2d_fwd_mean_var(
    const std::vector<float> &mu_w, const std::vector<float> &var_w,
    const std::vector<float> &mu_b, const std::vector<float> &var_b,
    const std::vector<float> &mu_a, const std::vector<float> &var_a,
    const std::vector<float> &mu_ra, const std::vector<float> &var_ra,
    bool bias, float epsilon, int wihi, int k, int start_chunk, int end_chunk,
    std::vector<float> &mu_z, std::vector<float> &var_z)
/*
 */
{
    for (int row = start_chunk; row < end_chunk; row++) {
        float inv_sqrt_var_ra = 1.0f / powf(var_ra[row] + epsilon, 0.5);
        float mu_ra_term = mu_ra[row];
        for (int col = 0; col < k; col++) {
            int idx = col + row * k;
            int idx_div = col / wihi;
            float mu_w_term = mu_w[idx_div];
            float mu_a_tilde = mu_a[idx] - mu_ra_term;

            mu_z[idx] = inv_sqrt_var_ra * mu_a_tilde * mu_w_term;
            var_z[idx] =
                inv_sqrt_var_ra * inv_sqrt_var_ra *
                (var_a[idx] * (mu_w_term * mu_w_term + var_w[idx_div]) +
                 var_w[idx_div] * mu_a_tilde * mu_a_tilde);

            if (bias) {
                mu_z[idx] += mu_b[idx_div];
                var_z[idx] += var_b[idx_div];
            }
        }
    }
}
////////////////////////////////////////////////////////////////////////////////
// Layer Norm's backward
////////////////////////////////////////////////////////////////////////////////
void layernorm_bwd_delta_z(const std::vector<float> &mu_w,
                           const std::vector<float> &jcb,
                           const std::vector<float> &var_ra,
                           const std::vector<float> &delta_mu_out,
                           const std::vector<float> &delta_var_out,
                           float epsilon, int ni, int start_chunk,
                           int end_chunk, std::vector<float> &delta_mu,
                           std::vector<float> &delta_var)
/*
 */
{
    for (int row = start_chunk; row < end_chunk; row++) {
        float inv_sqrt_var_ra = 1.0f / powf(var_ra[row] + epsilon, 0.5);
        for (int col = 0; col < ni; col++) {
            float tmp = inv_sqrt_var_ra * mu_w[col] * jcb[col + row * ni];

            delta_mu[col + row * ni] = tmp * delta_mu_out[col + row * ni];
            delta_var[col + row * ni] =
                tmp * delta_var_out[col + row * ni] * tmp;
        }
    }
}

void layernorm_bwd_delta_w(
    const std::vector<float> &var_w, const std::vector<float> &mu_a,
    const std::vector<float> &mu_ra, const std::vector<float> &var_ra,
    const std::vector<float> &delta_mu_out,
    const std::vector<float> &delta_var_out, float epsilon, int ni,
    int batch_size, int start_chunk, int end_chunk,
    std::vector<float> &delta_mu_w, std::vector<float> &delta_var_w)
/*
 */
{
    for (int col = start_chunk; col < end_chunk; col++) {
        float sum_mu = 0.0f;
        float sum_var = 0.0f;
        for (int row = 0; row < batch_size; row++) {
            float tmp = (1.0f / std::sqrt(var_ra[row] + epsilon)) *
                        (mu_a[col + row * ni] - mu_ra[row]) * var_w[col];

            sum_mu += tmp * delta_mu_out[col + row * ni];
            sum_var += tmp * delta_var_out[col + row * ni] * tmp;
        }
        delta_mu_w[col] = sum_mu;
        delta_var_w[col] = sum_var;
    }
}

void layernorm_bwd_delta_b(const std::vector<float> &var_b,
                           const std::vector<float> &delta_mu_out,
                           const std::vector<float> &delta_var_out,
                           float epsilon, int ni, int batch_size,
                           int start_chunk, int end_chunk,
                           std::vector<float> &delta_mu_b,
                           std::vector<float> &delta_var_b)
/*
 */
{
    for (int col = start_chunk; col < end_chunk; col++) {
        float sum_mu = 0.0f;
        float sum_var = 0.0f;
        for (int i = 0; i < batch_size; i++) {
            float A = var_b[col];
            sum_mu += A * delta_mu_out[col + i * ni];
            sum_var += A * delta_var_out[col + i * ni] * A;
        }
        delta_mu_b[col] = sum_mu;
        delta_var_b[col] = sum_var;
    }
}

void layernorm2d_bwd_delta_z(const std::vector<float> &mu_w,
                             const std::vector<float> &jcb,
                             const std::vector<float> &var_ra,
                             const std::vector<float> &delta_mu_out,
                             const std::vector<float> &delta_var_out,
                             float epsilon, int wihi, int fi, int start_chunk,
                             int end_chunk, std::vector<float> &delta_mu,
                             std::vector<float> &delta_var)
/*
 */
{
    // k = wihi * fi, m = B
    int k = wihi * fi;
    for (int row = start_chunk; row < end_chunk; row++) {
        float inv_sqrt_var_ra = 1.0f / powf(var_ra[row] + epsilon, 0.5);
        for (int col = 0; col < wihi * fi; col++) {
            float tmp = inv_sqrt_var_ra * mu_w[col / wihi] * jcb[col + row * k];

            delta_mu[col + row * k] = tmp * delta_mu_out[col + row * k];
            delta_var[col + row * k] = tmp * delta_var_out[col + row * k] * tmp;
        }
    }
}

void layernorm2d_bwd_delta_w(const std::vector<float> &var_w,
                             const std::vector<float> &mu_a,
                             const std::vector<float> &mu_ra,
                             const std::vector<float> &var_ra,
                             const std::vector<float> &delta_mu_out,
                             const std::vector<float> &delta_var_out,
                             float epsilon, int wihi, int fi, int start_chunk,
                             int end_chunk, std::vector<float> &delta_mu_w,
                             std::vector<float> &delta_var_w)
/*
 */
{
    // k = wihi, m = B
    int k = wihi * fi;
    for (int row = start_chunk; row < end_chunk; row++) {
        float inv_sqrt_var_ra = 1.0f / powf(var_ra[row] + epsilon, 0.5);
        for (int col = 0; col < k; col++) {
            float tmp = inv_sqrt_var_ra * (mu_a[col + row * k] - mu_ra[row]) *
                        var_w[col / wihi];

            delta_mu_w[col + row * k] = tmp * delta_mu_out[col + row * k];
            delta_var_w[col + row * k] =
                tmp * delta_var_out[col + row * k] * tmp;
        }
    }
}
void layernorm2d_bwd_delta_b(const std::vector<float> &var_b,
                             const std::vector<float> &delta_mu_out,
                             const std::vector<float> &delta_var_out,
                             float epsilon, int wihi, int fi, int start_chunk,
                             int end_chunk, std::vector<float> &delta_mu_b,
                             std::vector<float> &delta_var_b)
/*
 */
{
    // k = wihi*fi, m = B
    int k = wihi * fi;
    for (int row = start_chunk; row < end_chunk; row++) {
        for (int col = 0; col < k; col++) {
            float A = var_b[col / wihi];
            delta_mu_b[col + row * k] = A * delta_mu_out[col + row * k];
            delta_var_b[col + row * k] = A * delta_var_out[col + row * k] * A;
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// Multiprocessing kernels for layer norm
////////////////////////////////////////////////////////////////////////////////
void layernorm_stat_mean_var_mp(const std::vector<float> &mu_a,
                                const std::vector<float> &var_a, int ni,
                                int batch_size, const int num_threads,
                                std::vector<float> &mu_s,
                                std::vector<float> &var_s)
/*
 */
{
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    int n_per_thread = batch_size / num_threads;
    int extra = batch_size % num_threads;

    for (int i = 0; i < num_threads; i++) {
        int start_chunk = i * n_per_thread + std::min(i, extra);
        int end_chunk = start_chunk + n_per_thread + (i < extra ? 1 : 0);

        threads.emplace_back([=, &mu_a, &var_a, &mu_s, &var_s] {
            layernorm_stat_mean_var(mu_a, var_a, ni, start_chunk, end_chunk,
                                    mu_s, var_s);
        });
    }

    for (auto &thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
}

void layernorm_sample_var_mp(const std::vector<float> &mu_a,
                             const std::vector<float> &mu_s,
                             const std::vector<float> &var_s, int ni,
                             int batch_size, const int num_threads,
                             std::vector<float> &var_sample)
/*
 */
{
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    int n_per_thread = batch_size / num_threads;
    int extra = batch_size % num_threads;

    for (int i = 0; i < num_threads; i++) {
        int start_chunk = i * n_per_thread + std::min(i, extra);
        int end_chunk = start_chunk + n_per_thread + (i < extra ? 1 : 0);

        threads.emplace_back([=, &mu_a, &mu_s, &var_s, &var_sample] {
            layernorm_sample_var(mu_a, mu_s, var_s, ni, start_chunk, end_chunk,
                                 var_sample);
        });
    }

    for (auto &thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
}

void layernorm_fwd_mean_var_mp(
    const std::vector<float> &mu_w, const std::vector<float> &var_w,
    const std::vector<float> &mu_b, const std::vector<float> &var_b,
    const std::vector<float> &mu_a, const std::vector<float> &var_a,
    const std::vector<float> &mu_ra, const std::vector<float> &var_ra,
    bool bias, float epsilon, int ni, int batch_size, const int num_threads,
    std::vector<float> &mu_z, std::vector<float> &var_z)
/*
 */
{
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    int n_per_thread = batch_size / num_threads;
    int extra = batch_size % num_threads;

    for (int i = 0; i < num_threads; i++) {
        int start_chunk = i * n_per_thread + std::min(i, extra);
        int end_chunk = start_chunk + n_per_thread + (i < extra ? 1 : 0);

        threads.emplace_back([=, &mu_w, &var_w, &mu_b, &var_b, &mu_a, &var_a,
                              &mu_ra, &var_ra, &mu_z, &var_z] {
            layernorm_fwd_mean_var(mu_w, var_w, mu_b, var_b, mu_a, var_a, mu_ra,
                                   var_ra, bias, epsilon, ni, start_chunk,
                                   end_chunk, mu_z, var_z);
        });
    }

    for (auto &thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
}

void layernorm2d_fwd_mean_var_mp(
    const std::vector<float> &mu_w, const std::vector<float> &var_w,
    const std::vector<float> &mu_b, const std::vector<float> &var_b,
    const std::vector<float> &mu_a, const std::vector<float> &var_a,
    const std::vector<float> &mu_ra, const std::vector<float> &var_ra,
    bool bias, float epsilon, int wihi, int batch_size, int k,
    const int num_threads, std::vector<float> &mu_z, std::vector<float> &var_z)
/*
 */
{
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    int n_per_thread = batch_size / num_threads;
    int extra = batch_size % num_threads;

    for (int i = 0; i < num_threads; i++) {
        int start_chunk = i * n_per_thread + std::min(i, extra);
        int end_chunk = start_chunk + n_per_thread + (i < extra ? 1 : 0);

        threads.emplace_back([=, &mu_w, &var_w, &mu_b, &var_b, &mu_a, &var_a,
                              &mu_ra, &var_ra, &mu_z, &var_z] {
            layernorm2d_fwd_mean_var(mu_w, var_w, mu_b, var_b, mu_a, var_a,
                                     mu_ra, var_ra, bias, epsilon, wihi, k,
                                     start_chunk, end_chunk, mu_z, var_z);
        });
    }

    for (auto &thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
}

void layernorm_bwd_delta_z_mp(
    const std::vector<float> &mu_w, const std::vector<float> &jcb,
    const std::vector<float> &var_ra, const std::vector<float> &delta_mu_out,
    const std::vector<float> &delta_var_out, float epsilon, int ni,
    int batch_size, const int num_threads, std::vector<float> &delta_mu,
    std::vector<float> &delta_var)
/*
 */
{
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    int n_per_thread = batch_size / num_threads;
    int extra = batch_size % num_threads;

    for (int i = 0; i < num_threads; i++) {
        int start_chunk = i * n_per_thread + std::min(i, extra);
        int end_chunk = start_chunk + n_per_thread + (i < extra ? 1 : 0);

        threads.emplace_back([=, &mu_w, &jcb, &var_ra, &delta_mu_out,
                              &delta_var_out, &delta_mu, &delta_var] {
            layernorm_bwd_delta_z(mu_w, jcb, var_ra, delta_mu_out,
                                  delta_var_out, epsilon, ni, start_chunk,
                                  end_chunk, delta_mu, delta_var);
        });
    }

    for (auto &thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
}

void layernorm_bwd_delta_w_mp(
    const std::vector<float> &var_w, const std::vector<float> &mu_a,
    const std::vector<float> &mu_ra, const std::vector<float> &var_ra,
    const std::vector<float> &delta_mu_out,
    const std::vector<float> &delta_var_out, float epsilon, int ni,
    int batch_size, const int num_threads, std::vector<float> &delta_mu_w,
    std::vector<float> &delta_var_w)
/*
 */
{
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    int n_per_thread = ni / num_threads;
    int extra = ni % num_threads;

    for (int i = 0; i < num_threads; i++) {
        int start_chunk = i * n_per_thread + std::min(i, extra);
        int end_chunk = start_chunk + n_per_thread + (i < extra ? 1 : 0);

        threads.emplace_back([=, &var_w, &mu_a, &mu_ra, &var_ra, &delta_mu_out,
                              &delta_var_out, &delta_mu_w, &delta_var_w] {
            layernorm_bwd_delta_w(var_w, mu_a, mu_ra, var_ra, delta_mu_out,
                                  delta_var_out, epsilon, ni, batch_size,
                                  start_chunk, end_chunk, delta_mu_w,
                                  delta_var_w);
        });
    }

    for (auto &thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
}

void layernorm_bwd_delta_b_mp(const std::vector<float> &var_b,
                              const std::vector<float> &delta_mu_out,
                              const std::vector<float> &delta_var_out,
                              float epsilon, int ni, int batch_size,
                              const int num_threads,
                              std::vector<float> &delta_mu_b,
                              std::vector<float> &delta_var_b)
/*
 */
{
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    int n_per_thread = ni / num_threads;
    int extra = ni % num_threads;

    for (int i = 0; i < num_threads; i++) {
        int start_chunk = i * n_per_thread + std::min(i, extra);
        int end_chunk = start_chunk + n_per_thread + (i < extra ? 1 : 0);

        threads.emplace_back([=, &var_b, &delta_mu_out, &delta_var_out,
                              &delta_mu_b, &delta_var_b] {
            layernorm_bwd_delta_b(var_b, delta_mu_out, delta_var_out, epsilon,
                                  ni, batch_size, start_chunk, end_chunk,
                                  delta_mu_b, delta_var_b);
        });
    }

    for (auto &thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
}

void layernorm2d_bwd_delta_z_mp(
    const std::vector<float> &mu_w, const std::vector<float> &jcb,
    const std::vector<float> &var_ra, const std::vector<float> &delta_mu_out,
    const std::vector<float> &delta_var_out, float epsilon, int wihi, int fi,
    int batch_size, const int num_threads, std::vector<float> &delta_mu,
    std::vector<float> &delta_var)
/*
 */
{
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    int n_per_thread = batch_size / num_threads;
    int extra = batch_size % num_threads;

    for (int i = 0; i < num_threads; i++) {
        int start_chunk = i * n_per_thread + std::min(i, extra);
        int end_chunk = start_chunk + n_per_thread + (i < extra ? 1 : 0);

        threads.emplace_back([=, &mu_w, &jcb, &var_ra, &delta_mu_out,
                              &delta_var_out, &delta_mu, &delta_var] {
            layernorm2d_bwd_delta_z(
                mu_w, jcb, var_ra, delta_mu_out, delta_var_out, epsilon, wihi,
                fi, start_chunk, end_chunk, delta_mu, delta_var);
        });
    }

    for (auto &thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
}

void layernorm2d_bwd_delta_w_mp(
    const std::vector<float> &var_w, const std::vector<float> &mu_a,
    const std::vector<float> &mu_ra, const std::vector<float> &var_ra,
    const std::vector<float> &delta_mu_out,
    const std::vector<float> &delta_var_out, float epsilon, int wihi, int fi,
    int batch_size, const int num_threads, std::vector<float> &delta_mu_w,
    std::vector<float> &delta_var_w)
/*
 */
{
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    int n_per_thread = batch_size / num_threads;
    int extra = batch_size % num_threads;

    for (int i = 0; i < num_threads; i++) {
        int start_chunk = i * n_per_thread + std::min(i, extra);
        int end_chunk = start_chunk + n_per_thread + (i < extra ? 1 : 0);

        threads.emplace_back([=, &var_w, &mu_a, &mu_ra, &var_ra, &delta_mu_out,
                              &delta_var_out, &delta_mu_w, &delta_var_w] {
            layernorm2d_bwd_delta_w(var_w, mu_a, mu_ra, var_ra, delta_mu_out,
                                    delta_var_out, epsilon, wihi, fi,
                                    start_chunk, end_chunk, delta_mu_w,
                                    delta_var_w);
        });
    }

    for (auto &thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
}

void layernorm2d_bwd_delta_b_mp(const std::vector<float> &var_b,
                                const std::vector<float> &delta_mu_out,
                                const std::vector<float> &delta_var_out,
                                float epsilon, int wihi, int fi, int batch_size,
                                const int num_threads,
                                std::vector<float> &delta_mu_b,
                                std::vector<float> &delta_var_b)
/*
 */
{
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    int n_per_thread = batch_size / num_threads;
    int extra = batch_size % num_threads;

    for (int i = 0; i < num_threads; i++) {
        int start_chunk = i * n_per_thread + std::min(i, extra);
        int end_chunk = start_chunk + n_per_thread + (i < extra ? 1 : 0);

        threads.emplace_back([=, &var_b, &delta_mu_out, &delta_var_out,
                              &delta_mu_b, &delta_var_b] {
            layernorm2d_bwd_delta_b(var_b, delta_mu_out, delta_var_out, epsilon,
                                    wihi, fi, start_chunk, end_chunk,
                                    delta_mu_b, delta_var_b);
        });
    }

    for (auto &thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
}

void delta_param_sum_mp(const std::vector<float> delta_mu_e,
                        const std::vector<float> delta_var_e, int wihi, int fi,
                        int n, const int num_threads,
                        std::vector<float> delta_mu,
                        std::vector<float> delta_var) {}

////////////////////////////////////////////////////////////////////////////////
// Layer Norm class
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// Utility functions
////////////////////////////////////////////////////////////////////////////////

std::tuple<int, int> get_number_params_layer_norm(
    const std::vector<int> &normalized_shape)
/*
 */
{
    int num_elements = normalized_shape.size();
    int num_weights, num_biases;
    if (num_elements == 1 || num_elements == 3) {
        num_weights = normalized_shape[0];
        num_biases = normalized_shape[0];
    } else {
        throw std::runtime_error(
            "Error in file: " + std::string(__FILE__) +
            " at line: " + std::to_string(__LINE__) +
            ". Normalized shape provided are not supported.");
    }
    return {num_weights, num_biases};
}

////////////////////////////////////////////////////////////////////////////////
// Layer Norm
////////////////////////////////////////////////////////////////////////////////

LayerNorm::LayerNorm(const std::vector<int> &normalized_shape, float eps,
                     bool bias)
    : normalized_shape(normalized_shape),
      epsilon(eps)
/*
 */
{
    this->bias = bias;
    this->init_weight_bias();
    if (this->training) {
        this->allocate_param_delta();
    }
    if (this->normalized_shape.size() == 1) {
        this->input_size = this->normalized_shape[0];
        this->output_size = normalized_shape[0];
    } else if (this->normalized_shape.size() == 3) {
        this->in_channels = this->normalized_shape[0];
        this->in_width = this->normalized_shape[1];
        this->in_height = this->normalized_shape[2];
        this->out_channels = this->normalized_shape[0];
        this->out_width = this->normalized_shape[1];
        this->out_height = this->normalized_shape[2];
        this->input_size = this->in_channels * this->in_width * this->in_height;
        this->output_size =
            this->out_channels * this->out_width * this->out_height;
    } else {
        throw std::runtime_error(
            "Error in file: " + std::string(__FILE__) +
            " at line: " + std::to_string(__LINE__) +
            ". Normalized shape provided are not supported.");
    }
}

LayerNorm::~LayerNorm()
/**/
{}

std::string LayerNorm::get_layer_info() const
/*
 */
{
    return "LayerNorm()";
}

std::string LayerNorm::get_layer_name() const
/*
 */
{
    return "LayerNorm";
}

LayerType LayerNorm::get_layer_type() const
/*
 */
{
    return LayerType::Norm;
}

void LayerNorm::init_weight_bias()
/*
 */
{
    int num_features = this->normalized_shape[0];
    this->num_weights = this->normalized_shape[0];
    this->num_biases = this->bias ? this->normalized_shape[0] : 0;
    std::tie(this->mu_w, this->var_w, this->mu_b, this->var_b) =
        init_weight_bias_norm("", 1.0f, 1.0f, num_features, num_features,
                              this->num_weights, this->num_biases);
}

void LayerNorm::allocate_running_mean_var()
/*
 */
{
    this->mu_ra.resize(this->_batch_size, 0.0f);
    this->var_ra.resize(this->_batch_size, 1.0f);
}

void LayerNorm::forward(BaseHiddenStates &input_states,
                        BaseHiddenStates &output_states,
                        BaseTempStates &temp_states)
/**/
{
    // Checkout input size
    if (this->input_size != input_states.actual_size) {
        std::string message =
            "Input size mismatch: " + std::to_string(this->input_size) +
            " vs " + std::to_string(input_states.actual_size);
        LOG(LogLevel::ERROR, message);
    }

    int batch_size = input_states.block_size;
    if (this->_batch_size != batch_size) {
        this->_batch_size = batch_size;
        this->set_cap_factor_udapte(batch_size);
        this->allocate_running_mean_var();
    }

    // Assign output dimensions
    output_states.width = this->out_width;
    output_states.height = this->out_height;
    output_states.depth = this->out_channels;
    output_states.block_size = batch_size;
    output_states.actual_size = this->output_size;

    if (this->num_threads <= 1) {
        layernorm_stat_mean_var(input_states.mu_a, input_states.var_a,
                                this->input_size, 0, batch_size, this->mu_ra,
                                temp_states.tmp_2);

        layernorm_sample_var(input_states.mu_a, this->mu_ra, temp_states.tmp_2,
                             this->input_size, 0, batch_size, this->var_ra);

        if (this->normalized_shape.size() == 1) {
            layernorm_fwd_mean_var(
                this->mu_w, this->var_w, this->mu_b, this->var_b,
                input_states.mu_a, input_states.var_a, this->mu_ra,
                this->var_ra, this->bias, this->epsilon, this->input_size, 0,
                batch_size, output_states.mu_a, output_states.var_a);
        } else {
            int wihi = this->in_height * this->in_width;
            layernorm2d_fwd_mean_var(
                this->mu_w, this->var_w, this->mu_b, this->var_b,
                input_states.mu_a, input_states.var_a, this->mu_ra,
                this->var_ra, this->bias, this->epsilon, wihi, this->input_size,
                0, batch_size, output_states.mu_a, output_states.var_a);
        }
    } else {
        layernorm_stat_mean_var_mp(
            input_states.mu_a, input_states.var_a, this->input_size, batch_size,
            this->num_threads, this->mu_ra, temp_states.tmp_2);

        layernorm_sample_var_mp(input_states.mu_a, this->mu_ra,
                                temp_states.tmp_2, this->input_size, batch_size,
                                this->num_threads, this->var_ra);

        if (this->normalized_shape.size() == 1) {
            layernorm_fwd_mean_var_mp(
                this->mu_w, this->var_w, this->mu_b, this->var_b,
                input_states.mu_a, input_states.var_a, this->mu_ra,
                this->var_ra, this->bias, this->epsilon, this->input_size,
                batch_size, this->num_threads, output_states.mu_a,
                output_states.var_a);
        } else {
            int wihi = this->in_height * this->in_width;
            layernorm2d_fwd_mean_var_mp(
                this->mu_w, this->var_w, this->mu_b, this->var_b,
                input_states.mu_a, input_states.var_a, this->mu_ra,
                this->var_ra, this->bias, this->epsilon, wihi, batch_size,
                this->input_size, this->num_threads, output_states.mu_a,
                output_states.var_a);
        }
    }

    if (this->training) {
        this->storing_states_for_training(input_states, output_states);
    }
}

void LayerNorm::backward(BaseDeltaStates &input_delta_states,
                         BaseDeltaStates &output_delta_states,
                         BaseTempStates &temp_states, bool state_udapte)
/*
 */
{
    int batch_size = input_delta_states.block_size;

    if (state_udapte) {
        if (this->num_threads <= 1) {
            if (this->normalized_shape.size() == 1) {
                layernorm_bwd_delta_z(this->mu_w, this->bwd_states->jcb,
                                      this->var_ra, input_delta_states.delta_mu,
                                      input_delta_states.delta_var,
                                      this->epsilon, this->input_size, 0,
                                      batch_size, output_delta_states.delta_mu,
                                      output_delta_states.delta_var);
            } else {
                int wihi = this->in_height * this->in_width;

                layernorm2d_bwd_delta_z(
                    this->mu_w, this->bwd_states->jcb, this->var_ra,
                    input_delta_states.delta_mu, input_delta_states.delta_var,
                    this->epsilon, wihi, this->in_channels, 0, batch_size,
                    output_delta_states.delta_mu,
                    output_delta_states.delta_var);
            }
        } else {
            if (this->normalized_shape.size() == 1) {
                layernorm_bwd_delta_z_mp(
                    this->mu_w, this->bwd_states->jcb, this->var_ra,
                    input_delta_states.delta_mu, input_delta_states.delta_var,
                    this->epsilon, this->input_size, batch_size,
                    this->num_threads, output_delta_states.delta_mu,
                    output_delta_states.delta_var);
            } else {
                int wihi = this->in_height * this->in_width;

                layernorm2d_bwd_delta_z_mp(
                    this->mu_w, this->bwd_states->jcb, this->var_ra,
                    input_delta_states.delta_mu, input_delta_states.delta_var,
                    this->epsilon, wihi, this->in_channels, batch_size,
                    this->num_threads, output_delta_states.delta_mu,
                    output_delta_states.delta_var);
            }
        }
    }
    if (this->param_update) {
        if (this->num_threads <= 1) {
            if (this->normalized_shape.size() == 1) {
                layernorm_bwd_delta_w(
                    this->var_w, this->bwd_states->mu_a, this->mu_ra,
                    this->var_ra, input_delta_states.delta_mu,
                    input_delta_states.delta_var, this->epsilon,
                    this->input_size, batch_size, 0, this->input_size,
                    this->delta_mu_w, this->delta_var_w);

                if (this->bias) {
                    layernorm_bwd_delta_b(
                        this->var_b, input_delta_states.delta_mu,
                        input_delta_states.delta_var, this->epsilon,
                        this->input_size, batch_size, 0, this->input_size,
                        this->delta_mu_b, this->delta_var_b);
                }
            } else {
                int wihi = this->in_height * this->in_width;

                layernorm2d_bwd_delta_w(
                    this->var_w, this->bwd_states->mu_a, this->mu_ra,
                    this->var_ra, input_delta_states.delta_mu,
                    input_delta_states.delta_var, this->epsilon, wihi,
                    this->in_channels, 0, batch_size, temp_states.tmp_1,
                    temp_states.tmp_2);

                delta_param_sum(temp_states.tmp_1, temp_states.tmp_2, wihi,
                                this->in_channels, batch_size, this->delta_mu_w,
                                this->delta_var_w);

                if (this->bias) {
                    layernorm2d_bwd_delta_b(
                        this->var_b, input_delta_states.delta_mu,
                        input_delta_states.delta_var, this->epsilon, wihi,
                        this->in_channels, 0, batch_size, temp_states.tmp_1,
                        temp_states.tmp_2);

                    delta_param_sum(temp_states.tmp_1, temp_states.tmp_2, wihi,
                                    this->in_channels, batch_size,
                                    this->delta_mu_b, this->delta_var_b);
                }
            }
        } else {
            if (this->normalized_shape.size() == 1) {
                layernorm_bwd_delta_w_mp(
                    this->var_w, this->bwd_states->mu_a, this->mu_ra,
                    this->var_ra, input_delta_states.delta_mu,
                    input_delta_states.delta_var, this->epsilon,
                    this->input_size, batch_size, this->num_threads,
                    this->delta_mu_w, this->delta_var_w);

                if (this->bias) {
                    layernorm_bwd_delta_b_mp(
                        this->var_b, input_delta_states.delta_mu,
                        input_delta_states.delta_var, this->epsilon,
                        this->input_size, batch_size, this->num_threads,
                        this->delta_mu_b, this->delta_var_b);
                }
            } else {
                int wihi = this->in_height * this->in_width;

                layernorm2d_bwd_delta_w_mp(
                    this->var_w, this->bwd_states->mu_a, this->mu_ra,
                    this->var_ra, input_delta_states.delta_mu,
                    input_delta_states.delta_var, this->epsilon, wihi,
                    this->in_channels, batch_size, this->num_threads,
                    temp_states.tmp_1, temp_states.tmp_2);

                delta_param_sum(temp_states.tmp_1, temp_states.tmp_2, wihi,
                                this->in_channels, batch_size, this->delta_mu_w,
                                this->delta_var_w);

                if (this->bias) {
                    layernorm2d_bwd_delta_b_mp(
                        this->var_b, input_delta_states.delta_mu,
                        input_delta_states.delta_var, this->epsilon, wihi,
                        this->in_channels, batch_size, this->num_threads,
                        temp_states.tmp_1, temp_states.tmp_2);

                    delta_param_sum(temp_states.tmp_1, temp_states.tmp_2, wihi,
                                    this->in_channels, batch_size,
                                    this->delta_mu_b, this->delta_var_b);
                }
            }
        }
    }
}

#ifdef USE_CUDA
std::unique_ptr<BaseLayer> LayerNorm::to_cuda() {
    this->device = "cuda";
    return std::make_unique<LayerNormCuda>(this->normalized_shape,
                                           this->epsilon, this->bias);
}
#endif

std::tuple<std::vector<float>, std::vector<float>>
LayerNorm::get_running_mean_var()
/*
 */
{
    return {this->mu_ra, this->var_ra};
}

void LayerNorm::save(std::ofstream &file)
/*
 */
{
    if (!file.is_open()) {
        throw std::runtime_error("Error in file: " + std::string(__FILE__) +
                                 " at line: " + std::to_string(__LINE__) +
                                 ". Failed to open file for saving");
    }

    // Save the name length and name
    auto layer_name = this->get_layer_info();
    size_t name_length = layer_name.length();
    file.write(reinterpret_cast<char *>(&name_length), sizeof(name_length));
    file.write(layer_name.c_str(), name_length);

    for (const auto &m_w : this->mu_w) {
        file.write(reinterpret_cast<const char *>(&m_w), sizeof(m_w));
    }
    for (const auto &v_w : this->var_w) {
        file.write(reinterpret_cast<const char *>(&v_w), sizeof(v_w));
    }
    for (const auto &m_b : this->mu_b) {
        file.write(reinterpret_cast<const char *>(&m_b), sizeof(m_b));
    }
    for (const auto &v_b : this->var_b) {
        file.write(reinterpret_cast<const char *>(&v_b), sizeof(v_b));
    }

    // Running average for nomalization
    for (const auto &m_ra : this->mu_ra) {
        file.write(reinterpret_cast<const char *>(&m_ra), sizeof(m_ra));
    }
    for (const auto &v_ra : this->var_ra) {
        file.write(reinterpret_cast<const char *>(&v_ra), sizeof(v_ra));
    }
}

void LayerNorm::load(std::ifstream &file)
/*
 */
{
    if (!file.is_open()) {
        throw std::runtime_error("Error in file: " + std::string(__FILE__) +
                                 " at line: " + std::to_string(__LINE__) +
                                 ". Failed to open file for loading");
    }
    // Load the name length and name
    auto layer_name = this->get_layer_info();
    std::string loaded_name;
    size_t name_length;
    file.read(reinterpret_cast<char *>(&name_length), sizeof(name_length));
    loaded_name.resize(name_length);
    file.read(&loaded_name[0], name_length);

    // Check layer name
    if (layer_name != loaded_name) {
        throw std::runtime_error("Error in file: " + std::string(__FILE__) +
                                 " at line: " + std::to_string(__LINE__) +
                                 ". Layer name are not match. Expected: " +
                                 layer_name + ", Found: " + loaded_name);
    }

    for (auto &m_w : this->mu_w) {
        file.read(reinterpret_cast<char *>(&m_w), sizeof(m_w));
    }
    for (auto &v_w : this->var_w) {
        file.read(reinterpret_cast<char *>(&v_w), sizeof(v_w));
    }
    for (auto &m_b : this->mu_b) {
        file.read(reinterpret_cast<char *>(&m_b), sizeof(m_b));
    }
    for (auto &v_b : this->var_b) {
        file.read(reinterpret_cast<char *>(&v_b), sizeof(v_b));
    }

    // Running average for nomalization
    for (auto &m_ra : this->mu_ra) {
        file.read(reinterpret_cast<char *>(&m_ra), sizeof(m_ra));
    }
    for (auto &v_ra : this->var_ra) {
        file.read(reinterpret_cast<char *>(&v_ra), sizeof(v_ra));
    }

    this->num_weights = this->mu_w.size();
    this->num_biases = this->mu_b.size();
    if (this->training) {
        this->allocate_param_delta();
    }
}

std::tuple<std::vector<std::vector<float>>, std::vector<std::vector<float>>,
           std::vector<std::vector<float>>, std::vector<std::vector<float>>>
LayerNorm::get_norm_mean_var() {
    std::vector<std::vector<float>> mu_ras = {this->mu_ra};
    std::vector<std::vector<float>> var_ras = {this->var_ra};
    std::vector<std::vector<float>> mu_norms;
    std::vector<std::vector<float>> var_norms;
    return {mu_ras, var_ras, mu_norms, var_norms};
}
