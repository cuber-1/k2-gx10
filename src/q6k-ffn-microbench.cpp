#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-common.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

namespace {

constexpr int64_t HIDDEN = 8192;
constexpr int64_t INTERMEDIATE = 28672;
constexpr int64_t QK = 256;
constexpr uint64_t HARD_LIMIT = 2ULL * 1024 * 1024 * 1024;
constexpr uint64_t BACKEND_RESERVE = 256ULL * 1024 * 1024;
constexpr uint64_t METADATA_ALLOWANCE = 2ULL * 1024 * 1024;
constexpr double MAX_NMSE = 5e-4;
constexpr const char * OUTPUT_PREFIX =
    "/home/dvijraicha/k2-gx10/results/q6k-decode-q6-soa/outputs/";
constexpr const char * FULL_OUTPUT_PREFIX =
    "/home/dvijraicha/k2-gx10/results/q6k-decode-q6-full-soa/outputs/";

uint64_t checked_mul(uint64_t a, uint64_t b) {
    if (a != 0 && b > std::numeric_limits<uint64_t>::max()/a) {
        throw std::runtime_error("allocation-size multiplication overflow");
    }
    return a*b;
}

uint64_t checked_add(uint64_t a, uint64_t b) {
    if (b > std::numeric_limits<uint64_t>::max() - a) {
        throw std::runtime_error("allocation-size addition overflow");
    }
    return a + b;
}

struct Sizes {
    Sizes() {
        matrix = checked_mul(checked_mul(uint64_t(HIDDEN/QK), uint64_t(INTERMEDIATE)), sizeof(block_q6_K));
        weights = checked_mul(3, matrix);
        activations = checked_mul(uint64_t(HIDDEN + 2*INTERMEDIATE + HIDDEN), sizeof(float));
        tensor_buffer = checked_add(weights, activations);
        host_input = checked_mul(HIDDEN, sizeof(float));
        host_outputs = checked_mul(2*HIDDEN, sizeof(float));
        guarded_total = checked_add(
            checked_add(weights, tensor_buffer),
            checked_add(checked_add(host_input, host_outputs), BACKEND_RESERVE + METADATA_ALLOWANCE));
        guarded_total_repacked = checked_add(guarded_total, weights);
    }

    uint64_t matrix = 0;
    uint64_t weights = 0;
    uint64_t activations = 0;
    uint64_t tensor_buffer = 0;
    uint64_t host_input = 0;
    uint64_t host_outputs = 0;
    uint64_t guarded_total = 0;
    uint64_t guarded_total_repacked = 0;
};

void print_bytes(const char * label, uint64_t bytes) {
    std::printf("  %-28s %12llu B  %9.3f MiB\n", label,
        static_cast<unsigned long long>(bytes), double(bytes)/(1024.0*1024.0));
}

uint32_t mix32(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    return x ^ (x >> 16);
}

void initialize_matrix(uint8_t * data, size_t bytes, uint32_t seed) {
    auto * blocks = reinterpret_cast<block_q6_K *>(data);
    const size_t nblocks = bytes/sizeof(block_q6_K);
    for (size_t i = 0; i < nblocks; ++i) {
        block_q6_K & b = blocks[i];
        uint32_t state = mix32(static_cast<uint32_t>(i) ^ seed);
        for (uint8_t & v : b.ql) { state = mix32(state + 0x9e3779b9U); v = uint8_t(state); }
        for (uint8_t & v : b.qh) { state = mix32(state + 0x9e3779b9U); v = uint8_t(state); }
        for (int8_t & v : b.scales) {
            state = mix32(state + 0x9e3779b9U);
            v = static_cast<int8_t>(1 + state%7);
        }
        b.d = ggml_fp32_to_fp16(1.0f/512.0f);
    }
}

void initialize_data(std::vector<uint8_t> & weights, std::vector<float> & input, const Sizes & sizes) {
    initialize_matrix(weights.data() + 0*sizes.matrix, sizes.matrix, 0x4b325550U);
    initialize_matrix(weights.data() + 1*sizes.matrix, sizes.matrix, 0x4b324741U);
    initialize_matrix(weights.data() + 2*sizes.matrix, sizes.matrix, 0x4b32444eU);
    for (size_t i = 0; i < input.size(); ++i) {
        const int value = int(mix32(static_cast<uint32_t>(i) ^ 0x4b32494eU)%2001U) - 1000;
        input[i] = float(value)/1000.0f;
    }
}

void repack_q6_soa_matrix(const uint8_t * src, uint8_t * dst, int64_t ncols, int64_t nrows) {
    static_assert(sizeof(block_q6_K) == 210, "unexpected Q6_K block size");
    static_assert(offsetof(block_q6_K, ql) == 0, "unexpected Q6_K ql offset");
    static_assert(offsetof(block_q6_K, qh) == 128, "unexpected Q6_K qh offset");
    static_assert(offsetof(block_q6_K, scales) == 192, "unexpected Q6_K scales offset");
    static_assert(offsetof(block_q6_K, d) == 208, "unexpected Q6_K d offset");
    if ((ncols != HIDDEN && ncols != INTERMEDIATE) || ncols % QK != 0) {
        throw std::runtime_error("unsupported Q6 SoA matrix width");
    }
    const int64_t blocks_per_row = ncols/QK;
    const int64_t rows_per_group = ncols == HIDDEN ? 2 : 4;
    if (nrows % rows_per_group != 0) throw std::runtime_error("incomplete Q6 SoA row group");
    const int64_t blocks_per_group = rows_per_group*blocks_per_row;
    const size_t bytes_per_group = size_t(blocks_per_group)*sizeof(block_q6_K);
    if (bytes_per_group % 128 != 0) throw std::runtime_error("unaligned Q6 SoA group size");

    const auto * blocks = reinterpret_cast<const block_q6_K *>(src);
    for (int64_t group = 0; group < nrows/rows_per_group; ++group) {
        uint8_t * group_dst = dst + size_t(group)*bytes_per_group;
        uint8_t * ql_dst = group_dst;
        uint8_t * tail_dst = group_dst + size_t(blocks_per_group)*128;
        for (int64_t row_in_group = 0; row_in_group < rows_per_group; ++row_in_group) {
            const int64_t row = group*rows_per_group + row_in_group;
            for (int64_t kbx = 0; kbx < blocks_per_row; ++kbx) {
                const int64_t block_in_group = row_in_group*blocks_per_row + kbx;
                const block_q6_K & block = blocks[row*blocks_per_row + kbx];
                std::memcpy(ql_dst + size_t(block_in_group)*128, block.ql, 128);
                std::memcpy(tail_dst + size_t(block_in_group)*82, block.qh, 82);
            }
        }
    }
}

void verify_q6_soa_matrix(const uint8_t * original, const uint8_t * packed, int64_t ncols, int64_t nrows) {
    const int64_t blocks_per_row = ncols/QK;
    const int64_t rows_per_group = ncols == HIDDEN ? 2 : 4;
    const int64_t blocks_per_group = rows_per_group*blocks_per_row;
    const size_t bytes_per_group = size_t(blocks_per_group)*sizeof(block_q6_K);
    const auto * blocks = reinterpret_cast<const block_q6_K *>(original);
    for (int64_t group = 0; group < nrows/rows_per_group; ++group) {
        const uint8_t * group_src = packed + size_t(group)*bytes_per_group;
        const uint8_t * ql_src = group_src;
        const uint8_t * tail_src = group_src + size_t(blocks_per_group)*128;
        for (int64_t row_in_group = 0; row_in_group < rows_per_group; ++row_in_group) {
            const int64_t row = group*rows_per_group + row_in_group;
            for (int64_t kbx = 0; kbx < blocks_per_row; ++kbx) {
                const int64_t block_in_group = row_in_group*blocks_per_row + kbx;
                const block_q6_K & block = blocks[row*blocks_per_row + kbx];
                if (std::memcmp(ql_src + size_t(block_in_group)*128, block.ql, 128) != 0 ||
                        std::memcmp(tail_src + size_t(block_in_group)*82, block.qh, 82) != 0) {
                    throw std::runtime_error("Q6 SoA round-trip verification failed");
                }
            }
        }
    }
}

std::vector<uint8_t> repack_q6_soa_weights(const std::vector<uint8_t> & original, const Sizes & sizes) {
    std::vector<uint8_t> packed(original.size());
    repack_q6_soa_matrix(original.data() + 0*sizes.matrix, packed.data() + 0*sizes.matrix, HIDDEN, INTERMEDIATE);
    repack_q6_soa_matrix(original.data() + 1*sizes.matrix, packed.data() + 1*sizes.matrix, HIDDEN, INTERMEDIATE);
    repack_q6_soa_matrix(original.data() + 2*sizes.matrix, packed.data() + 2*sizes.matrix, INTERMEDIATE, HIDDEN);
    verify_q6_soa_matrix(original.data() + 0*sizes.matrix, packed.data() + 0*sizes.matrix, HIDDEN, INTERMEDIATE);
    verify_q6_soa_matrix(original.data() + 1*sizes.matrix, packed.data() + 1*sizes.matrix, HIDDEN, INTERMEDIATE);
    verify_q6_soa_matrix(original.data() + 2*sizes.matrix, packed.data() + 2*sizes.matrix, INTERMEDIATE, HIDDEN);
    return packed;
}

void repack_q6_full_soa_matrix(const uint8_t * src, uint8_t * dst, int64_t ncols, int64_t nrows) {
    const int64_t blocks_per_row = ncols/QK;
    const int64_t rows_per_group = ncols == HIDDEN ? 2 : 4;
    const int64_t blocks_per_group = rows_per_group*blocks_per_row;
    const size_t bytes_per_group = size_t(blocks_per_group)*sizeof(block_q6_K);
    if ((ncols != HIDDEN && ncols != INTERMEDIATE) || nrows % rows_per_group != 0 || bytes_per_group % 128 != 0) {
        throw std::runtime_error("unsupported full Q6 SoA matrix shape");
    }
    const auto * blocks = reinterpret_cast<const block_q6_K *>(src);
    for (int64_t group = 0; group < nrows/rows_per_group; ++group) {
        uint8_t * group_dst = dst + size_t(group)*bytes_per_group;
        uint8_t * ql_dst = group_dst;
        uint8_t * qh_dst = ql_dst + size_t(blocks_per_group)*128;
        uint8_t * scales_dst = qh_dst + size_t(blocks_per_group)*64;
        uint8_t * d_dst = scales_dst + size_t(blocks_per_group)*16;
        for (int64_t row_in_group = 0; row_in_group < rows_per_group; ++row_in_group) {
            const int64_t row = group*rows_per_group + row_in_group;
            for (int64_t kbx = 0; kbx < blocks_per_row; ++kbx) {
                const int64_t block_in_group = row_in_group*blocks_per_row + kbx;
                const block_q6_K & block = blocks[row*blocks_per_row + kbx];
                std::memcpy(ql_dst + size_t(block_in_group)*128, block.ql, 128);
                std::memcpy(qh_dst + size_t(block_in_group)*64, block.qh, 64);
                std::memcpy(scales_dst + size_t(block_in_group)*16, block.scales, 16);
                std::memcpy(d_dst + size_t(block_in_group)*2, &block.d, 2);
            }
        }
    }
}

void verify_q6_full_soa_matrix(const uint8_t * original, const uint8_t * packed, int64_t ncols, int64_t nrows) {
    const int64_t blocks_per_row = ncols/QK;
    const int64_t rows_per_group = ncols == HIDDEN ? 2 : 4;
    const int64_t blocks_per_group = rows_per_group*blocks_per_row;
    const size_t bytes_per_group = size_t(blocks_per_group)*sizeof(block_q6_K);
    const auto * blocks = reinterpret_cast<const block_q6_K *>(original);
    for (int64_t group = 0; group < nrows/rows_per_group; ++group) {
        const uint8_t * group_src = packed + size_t(group)*bytes_per_group;
        const uint8_t * ql_src = group_src;
        const uint8_t * qh_src = ql_src + size_t(blocks_per_group)*128;
        const uint8_t * scales_src = qh_src + size_t(blocks_per_group)*64;
        const uint8_t * d_src = scales_src + size_t(blocks_per_group)*16;
        for (int64_t row_in_group = 0; row_in_group < rows_per_group; ++row_in_group) {
            const int64_t row = group*rows_per_group + row_in_group;
            for (int64_t kbx = 0; kbx < blocks_per_row; ++kbx) {
                const int64_t block_in_group = row_in_group*blocks_per_row + kbx;
                const block_q6_K & block = blocks[row*blocks_per_row + kbx];
                if (std::memcmp(ql_src + size_t(block_in_group)*128, block.ql, 128) != 0 ||
                        std::memcmp(qh_src + size_t(block_in_group)*64, block.qh, 64) != 0 ||
                        std::memcmp(scales_src + size_t(block_in_group)*16, block.scales, 16) != 0 ||
                        std::memcmp(d_src + size_t(block_in_group)*2, &block.d, 2) != 0) {
                    throw std::runtime_error("full Q6 SoA round-trip verification failed");
                }
            }
        }
    }
}

std::vector<uint8_t> repack_q6_full_soa_weights(const std::vector<uint8_t> & original, const Sizes & sizes) {
    std::vector<uint8_t> packed(original.size());
    repack_q6_full_soa_matrix(original.data() + 0*sizes.matrix, packed.data() + 0*sizes.matrix, HIDDEN, INTERMEDIATE);
    repack_q6_full_soa_matrix(original.data() + 1*sizes.matrix, packed.data() + 1*sizes.matrix, HIDDEN, INTERMEDIATE);
    repack_q6_full_soa_matrix(original.data() + 2*sizes.matrix, packed.data() + 2*sizes.matrix, INTERMEDIATE, HIDDEN);
    verify_q6_full_soa_matrix(original.data() + 0*sizes.matrix, packed.data() + 0*sizes.matrix, HIDDEN, INTERMEDIATE);
    verify_q6_full_soa_matrix(original.data() + 1*sizes.matrix, packed.data() + 1*sizes.matrix, HIDDEN, INTERMEDIATE);
    verify_q6_full_soa_matrix(original.data() + 2*sizes.matrix, packed.data() + 2*sizes.matrix, INTERMEDIATE, HIDDEN);
    return packed;
}

struct RunResult {
    std::vector<float> output;
    double median_ms = 0.0;
    double minimum_ms = 0.0;
    double maximum_ms = 0.0;
    size_t buffer_bytes = 0;
};

RunResult run_backend(ggml_backend_t backend, const std::vector<uint8_t> & weights,
        const std::vector<float> & input, const Sizes & sizes, bool cpu, int warmup, int iterations) {
    ggml_init_params params = { METADATA_ALLOWANCE, nullptr, true };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) throw std::runtime_error("ggml_init failed");

    ggml_tensor * w_up = ggml_new_tensor_2d(ctx, GGML_TYPE_Q6_K, HIDDEN, INTERMEDIATE);
    ggml_tensor * w_gate = ggml_new_tensor_2d(ctx, GGML_TYPE_Q6_K, HIDDEN, INTERMEDIATE);
    ggml_tensor * w_down = ggml_new_tensor_2d(ctx, GGML_TYPE_Q6_K, INTERMEDIATE, HIDDEN);
    ggml_tensor * x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, HIDDEN);
    ggml_tensor * up = ggml_mul_mat(ctx, w_up, x);
    ggml_tensor * gate = ggml_mul_mat(ctx, w_gate, x);
    ggml_tensor * glu = ggml_swiglu_split(ctx, gate, up);
    ggml_tensor * y = ggml_mul_mat(ctx, w_down, glu);

    ggml_set_name(w_up, "synthetic_q6k_up_weight");
    ggml_set_name(w_gate, "synthetic_q6k_gate_weight");
    ggml_set_name(w_down, "synthetic_q6k_down_weight");
    ggml_set_name(x, "deterministic_f32_input");
    ggml_set_name(up, "up_projection");
    ggml_set_name(gate, "gate_projection");
    ggml_set_name(glu, "swiglu_intermediate");
    ggml_set_name(y, "ffn_output");

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 16, false);
    ggml_build_forward_expand(graph, y);
    if (ggml_graph_n_nodes(graph) != 4) {
        ggml_free(ctx);
        throw std::runtime_error("unexpected FFN graph node count");
    }

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) {
        ggml_free(ctx);
        throw std::runtime_error("backend tensor-buffer allocation failed");
    }
    const size_t buffer_bytes = ggml_backend_buffer_get_size(buffer);
    if (buffer_bytes >= HARD_LIMIT) {
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        throw std::runtime_error("backend tensor buffer violates hard limit");
    }

    ggml_backend_tensor_set(w_up, weights.data() + 0*sizes.matrix, 0, sizes.matrix);
    ggml_backend_tensor_set(w_gate, weights.data() + 1*sizes.matrix, 0, sizes.matrix);
    ggml_backend_tensor_set(w_down, weights.data() + 2*sizes.matrix, 0, sizes.matrix);
    ggml_backend_tensor_set(x, input.data(), 0, input.size()*sizeof(float));
    if (cpu) {
        const unsigned hw = std::thread::hardware_concurrency();
        ggml_backend_cpu_set_n_threads(backend, std::max(1U, std::min(16U, hw)));
    }

    for (int i = 0; i < warmup; ++i) {
        const ggml_status status = ggml_backend_graph_compute(backend, graph);
        ggml_backend_synchronize(backend);
        if (status != GGML_STATUS_SUCCESS) {
            ggml_backend_buffer_free(buffer);
            ggml_free(ctx);
            throw std::runtime_error(std::string("warmup failed: ") + ggml_status_to_string(status));
        }
    }

    std::vector<double> samples;
    samples.reserve(iterations);
    for (int i = 0; i < iterations; ++i) {
        const auto start = std::chrono::steady_clock::now();
        const ggml_status status = ggml_backend_graph_compute(backend, graph);
        ggml_backend_synchronize(backend);
        const auto stop = std::chrono::steady_clock::now();
        if (status != GGML_STATUS_SUCCESS) {
            ggml_backend_buffer_free(buffer);
            ggml_free(ctx);
            throw std::runtime_error(std::string("compute failed: ") + ggml_status_to_string(status));
        }
        samples.push_back(std::chrono::duration<double, std::milli>(stop - start).count());
    }
    std::sort(samples.begin(), samples.end());

    RunResult result;
    result.output.resize(HIDDEN);
    ggml_backend_tensor_get(y, result.output.data(), 0, result.output.size()*sizeof(float));
    result.median_ms = samples[samples.size()/2];
    result.minimum_ms = samples.front();
    result.maximum_ms = samples.back();
    result.buffer_bytes = buffer_bytes;
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return result;
}

double nmse(const std::vector<float> & reference, const std::vector<float> & actual, double & max_abs) {
    long double error2 = 0.0;
    long double reference2 = 0.0;
    max_abs = 0.0;
    for (size_t i = 0; i < reference.size(); ++i) {
        if (!std::isfinite(reference[i]) || !std::isfinite(actual[i])) {
            return std::numeric_limits<double>::infinity();
        }
        const double delta = double(reference[i]) - actual[i];
        error2 += delta*delta;
        reference2 += double(reference[i])*reference[i];
        max_abs = std::max(max_abs, std::abs(delta));
    }
    return reference2 == 0.0 ? double(error2) : double(error2/reference2);
}

} // namespace

int main(int argc, char ** argv) try {
    bool execute = false;
    bool skip_cpu = false;
    bool repacked = false;
    bool repacked_full = false;
    bool allow_root_profile = false;
    int warmup = 0;
    int iterations = 1;
    std::string gpu_output_path;
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "--execute") execute = true;
        else if (arg == "--dry-run") execute = false;
        else if (arg == "--skip-cpu") skip_cpu = true;
        else if (arg == "--repacked") repacked = true;
        else if (arg == "--repacked-full") repacked_full = true;
        else if (arg == "--allow-root-profile") allow_root_profile = true;
        else if (arg == "--warmup" && i + 1 < argc) warmup = std::stoi(argv[++i]);
        else if (arg == "--iterations" && i + 1 < argc) iterations = std::stoi(argv[++i]);
        else if (arg == "--gpu-output" && i + 1 < argc) gpu_output_path = argv[++i];
        else throw std::runtime_error("accepted options: --dry-run, --execute, --skip-cpu, --repacked, --repacked-full, --warmup N, "
                                      "--iterations N, --gpu-output PATH, --allow-root-profile");
    }
    if (warmup < 0 || warmup > 20) throw std::runtime_error("--warmup must be in 0..20");
    if (iterations < 1 || iterations > 100) throw std::runtime_error("--iterations must be in 1..100");
    if (repacked && repacked_full) throw std::runtime_error("select only one repacked layout");
    if (geteuid() == 0 && !allow_root_profile) {
        throw std::runtime_error("refusing root execution without --allow-root-profile");
    }
    if (geteuid() != 0 && allow_root_profile) {
        throw std::runtime_error("--allow-root-profile is reserved for privileged profiling");
    }
    if (!gpu_output_path.empty()) {
        const std::string soa_prefix(OUTPUT_PREFIX);
        const std::string full_prefix(FULL_OUTPUT_PREFIX);
        const bool soa_path = gpu_output_path.compare(0, soa_prefix.size(), soa_prefix) == 0;
        const bool full_path = gpu_output_path.compare(0, full_prefix.size(), full_prefix) == 0;
        const bool allowed_path = repacked ? soa_path : (repacked_full ? full_path : (soa_path || full_path));
        if (!execute || !allowed_path ||
                gpu_output_path.find("..") != std::string::npos) {
            throw std::runtime_error("--gpu-output is restricted to the experiment outputs directory");
        }
        if (access(gpu_output_path.c_str(), F_OK) == 0) {
            throw std::runtime_error("--gpu-output refuses to overwrite an existing path");
        }
    }

    const Sizes sizes;
    std::puts("K2 exact Q6_K decode FFN microbenchmark");
    std::puts("shape: F32[8192] -> Q6_K up+gate[28672,8192] -> SwiGLU -> Q6_K down[8192,28672]");
    std::printf("mode: %s; cpu_reference=%s; layout=%s; warmup=%d; iterations=%d\n",
        execute ? "execute" : "dry-run", skip_cpu ? "no" : "yes",
        repacked_full ? "q6-full-field-soa" : (repacked ? "q6-field-soa" : "row-major"), warmup, iterations);
    print_bytes("one Q6_K matrix", sizes.matrix);
    print_bytes("three Q6_K matrices", sizes.weights);
    print_bytes("one backend tensor buffer", sizes.tensor_buffer);
    print_bytes("guarded total", (repacked || repacked_full) ? sizes.guarded_total_repacked : sizes.guarded_total);
    print_bytes("hard limit", HARD_LIMIT);
    if (((repacked || repacked_full) ? sizes.guarded_total_repacked : sizes.guarded_total) >= HARD_LIMIT) {
        throw std::runtime_error("guarded total reaches 2 GiB");
    }
    if (!execute) {
        std::puts("DRY RUN PASS: no backend initialized and no GPU memory allocated");
        return 0;
    }

    std::vector<uint8_t> weights(sizes.weights);
    std::vector<float> input(HIDDEN);
    initialize_data(weights, input, sizes);

    std::vector<uint8_t> packed_weights;
    if (repacked_full) {
        packed_weights = repack_q6_full_soa_weights(weights, sizes);
        std::puts("Q6 full SoA exact-size repack + round-trip: PASS");
    } else if (repacked) {
        packed_weights = repack_q6_soa_weights(weights, sizes);
        std::puts("Q6 SoA exact-size repack + round-trip: PASS");
    }

    RunResult reference;
    if (!skip_cpu) {
        ggml_backend_t cpu = ggml_backend_cpu_init();
        if (!cpu) throw std::runtime_error("CPU backend initialization failed");
        reference = run_backend(cpu, weights, input, sizes, true, 0, 1);
        ggml_backend_free(cpu);
        std::printf("CPU reference: %.6f ms\n", reference.median_ms);
    }

    if (ggml_backend_cuda_get_device_count() < 1) throw std::runtime_error("no CUDA device found");
    if (repacked_full) {
        if (unsetenv("K2_Q6_SOA") != 0 || setenv("K2_Q6_FULL_SOA", "1", 1) != 0) {
            throw std::runtime_error("failed to enable K2_Q6_FULL_SOA");
        }
    } else if (repacked) {
        if (unsetenv("K2_Q6_FULL_SOA") != 0) throw std::runtime_error("failed to clear K2_Q6_FULL_SOA");
        if (setenv("K2_Q6_SOA", "1", 1) != 0) throw std::runtime_error("failed to enable K2_Q6_SOA");
    } else {
        if (unsetenv("K2_Q6_SOA") != 0 || unsetenv("K2_Q6_FULL_SOA") != 0) {
            throw std::runtime_error("failed to clear experimental layout environment");
        }
    }
    ggml_backend_t cuda = ggml_backend_cuda_init(0);
    if (!cuda) throw std::runtime_error("CUDA backend initialization failed");
    const std::vector<uint8_t> & gpu_weights = (repacked || repacked_full) ? packed_weights : weights;
    RunResult actual = run_backend(cuda, gpu_weights, input, sizes, false, warmup, iterations);
    ggml_backend_free(cuda);

    if (!gpu_output_path.empty()) {
        std::FILE * output = std::fopen(gpu_output_path.c_str(), "wb");
        if (!output) throw std::runtime_error("failed to open GPU output path");
        const size_t written = std::fwrite(actual.output.data(), sizeof(float), actual.output.size(), output);
        const int close_status = std::fclose(output);
        if (written != actual.output.size() || close_status != 0) {
            throw std::runtime_error("failed to write complete GPU output");
        }
    }

    std::printf("CUDA timing: median=%.6f ms min=%.6f ms max=%.6f ms buffer=%zu B\n",
        actual.median_ms, actual.minimum_ms, actual.maximum_ms, actual.buffer_bytes);
    if (!skip_cpu) {
        double max_abs = 0.0;
        const double error = nmse(reference.output, actual.output, max_abs);
        std::printf("correctness: NMSE=%.9g threshold=%.9g max_abs=%.9g\n", error, MAX_NMSE, max_abs);
        if (!(error <= MAX_NMSE)) throw std::runtime_error("CPU/CUDA correctness comparison failed");
    }
    std::puts("EXECUTION PASS");
    return 0;
} catch (const std::exception & e) {
    std::fprintf(stderr, "ERROR: %s\n", e.what());
    return 1;
}
