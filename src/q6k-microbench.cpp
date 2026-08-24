#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-common.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

namespace {

constexpr int64_t DEFAULT_K = 8192;
constexpr int64_t DEFAULT_M = 28672;
constexpr int64_t DEFAULT_N = 25;
constexpr int64_t MAX_N = 2048;
constexpr int64_t MAX_DIM = 28672;
constexpr int64_t QK = 256;
constexpr int64_t TILE_I = 128;
constexpr int64_t GB10_SMS = 48;
constexpr uint64_t Q8_1_MMQ_TILE_BYTES = 144;
constexpr uint64_t HARD_LIMIT = 2ULL * 1024 * 1024 * 1024;
constexpr uint64_t BACKEND_RESERVE = 256ULL * 1024 * 1024;
constexpr uint64_t METADATA_ALLOWANCE = 2ULL * 1024 * 1024;
constexpr double MAX_NMSE = 5e-4;

static_assert(sizeof(block_q6_K) == 210 || sizeof(block_q6_K) == 212,
              "Unsupported Q6_K experiment layout");

int64_t select_mmq_tile_j(int64_t n) {
    constexpr int64_t supported[] = {8, 16, 24, 32, 40, 48, 64, 80, 96, 112, 128};
    for (const int64_t j : supported) {
        if (n <= j) return j;
    }
    return 128;
}

uint64_t checked_mul(uint64_t a, uint64_t b) {
    if (a != 0 && b > std::numeric_limits<uint64_t>::max() / a) {
        throw std::runtime_error("allocation-size multiplication overflow");
    }
    return a * b;
}

uint64_t checked_sum(std::initializer_list<uint64_t> values) {
    uint64_t total = 0;
    for (const uint64_t value : values) {
        if (value > std::numeric_limits<uint64_t>::max() - total) {
            throw std::runtime_error("allocation-size addition overflow");
        }
        total += value;
    }
    return total;
}

struct Sizes {
    explicit Sizes(int64_t m, int64_t k, int64_t n, bool fused_swiglu = false) :
        weight(checked_mul(checked_mul(uint64_t(k / QK), uint64_t(m)), sizeof(block_q6_K))),
        input(checked_mul(checked_mul(uint64_t(k), uint64_t(n)), sizeof(float))),
        output(checked_mul(checked_mul(uint64_t(m), uint64_t(n)), sizeof(float))),
        tensor_buffer(checked_sum({checked_mul(fused_swiglu ? 2 : 1, weight), input,
                                   checked_mul(fused_swiglu ? 3 : 1, output)})),
        host_outputs(checked_mul(2, output)),
        init_scratch(checked_mul(uint64_t(k), sizeof(float))),
        q8_workspace(checked_sum({checked_mul(checked_mul(uint64_t(n), uint64_t(k / 32)), 36),
                                  checked_mul(uint64_t(select_mmq_tile_j(n)),
                                              Q8_1_MMQ_TILE_BYTES)})),
        stream_k_fixup(calc_stream_k_fixup(m, n)),
        guarded_total(checked_sum({checked_mul(2, tensor_buffer), weight, input, host_outputs,
                                   init_scratch, q8_workspace, stream_k_fixup,
                                   METADATA_ALLOWANCE, BACKEND_RESERVE})) {}

    static uint64_t calc_stream_k_fixup(int64_t m, int64_t n) {
        if (n <= 8) return 0;
        const int64_t j = select_mmq_tile_j(n);
        const int64_t ntiles = (m / TILE_I) * ((n + j - 1) / j);
        const int64_t nwaves = (ntiles + GB10_SMS - 1) / GB10_SMS;
        const int64_t efficiency = 100 * ntiles / (GB10_SMS * nwaves);
        const int64_t blocks = efficiency >= 90 ? ntiles : GB10_SMS;
        return ntiles % blocks == 0 ? 0 :
            checked_mul(checked_mul(checked_mul(uint64_t(blocks), uint64_t(j)), uint64_t(TILE_I)),
                        sizeof(float));
    }

    uint64_t weight;
    uint64_t input;
    uint64_t output;
    uint64_t tensor_buffer;
    uint64_t host_outputs;
    uint64_t init_scratch;
    uint64_t q8_workspace;
    uint64_t stream_k_fixup;
    uint64_t guarded_total;
};

void print_bytes(const char * label, uint64_t bytes) {
    std::printf("  %-34s %12llu B  %9.3f MiB\n", label,
                static_cast<unsigned long long>(bytes), double(bytes) / (1024.0 * 1024.0));
}

void print_plan(bool execute, bool fused_swiglu, int64_t m, int64_t k, int64_t n,
                int warmup, int iterations) {
    const Sizes s(m, k, n, fused_swiglu);
    const int64_t tile_j = select_mmq_tile_j(n);
    const int64_t nty = m / TILE_I;
    const int64_t ntx = (n + tile_j - 1) / tile_j;
    const int64_t ntiles = nty * ntx;
    const int64_t nwaves = (ntiles + GB10_SMS - 1) / GB10_SMS;
    const int64_t efficiency = 100 * ntiles / (GB10_SMS * nwaves);
    const int64_t input_bytes = k * n * int64_t(sizeof(float));
    const int64_t output_bytes = m * n * int64_t(sizeof(float));
    const uint64_t q6k_row_bytes = uint64_t(k / QK) * sizeof(block_q6_K);
    const uint64_t q6k_tensor_bytes = q6k_row_bytes * m;

    std::puts("Q6_K CUDA matrix-multiplication microbenchmark");
    std::printf("mode: %s\n", execute ? "execute" : "dry-run (use --execute to allocate or compute)");
    std::puts("model/GGUF loading: forbidden; no path accepted or loader called");
    std::puts("server startup: forbidden; llama server/model code is not linked");
    std::printf("columns: %lld (allowed range: 1..%lld)\n",
                static_cast<long long>(n), static_cast<long long>(MAX_N));
    std::printf("rows: %lld (multiple of %lld)\n",
                static_cast<long long>(m), static_cast<long long>(TILE_I));
    std::printf("k: %lld (multiple of %lld)\n",
                static_cast<long long>(k), static_cast<long long>(QK));
    std::printf("timing: warmup=%d iterations=%d\n", warmup, iterations);
    std::printf("decode fusion: %s\n", fused_swiglu ? "Q6_K up+gate SwiGLU" : "none");
    if (n <= 8) {
        std::printf("predicted MMVQ specialization: mul_mat_vec_q<(ggml_type)14, %lld, %s, false>\n",
                    static_cast<long long>(n), fused_swiglu ? "true" : "false");
    } else {
        std::printf("predicted MMQ specialization if selected: mul_mat_q<(ggml_type)14, %lld, false>\n",
                    static_cast<long long>(tile_j));
    }
    std::printf("src0 Q6_K: ne=[%lld,%lld,1,1] nb=[%zu,%llu,%llu,%llu]\n",
                static_cast<long long>(k), static_cast<long long>(m), sizeof(block_q6_K),
                static_cast<unsigned long long>(q6k_row_bytes),
                static_cast<unsigned long long>(q6k_tensor_bytes),
                static_cast<unsigned long long>(q6k_tensor_bytes));
    std::printf("src1 F32:  ne=[%lld,%lld,1,1] nb=[4,%lld,%lld,%lld]\n",
                static_cast<long long>(k), static_cast<long long>(n),
                static_cast<long long>(k * int64_t(sizeof(float))),
                static_cast<long long>(input_bytes), static_cast<long long>(input_bytes));
    std::printf("dst  F32:  ne=[%lld,%lld,1,1] nb=[4,%lld,%lld,%lld]\n",
                static_cast<long long>(m), static_cast<long long>(n),
                static_cast<long long>(m * int64_t(sizeof(float))),
                static_cast<long long>(output_bytes), static_cast<long long>(output_bytes));
    if (n <= 8) {
        const int64_t rows_per_block = n == 1 ? 1 : 2;
        const int64_t warps_per_block = n == 1 ? 8 : 4;
        std::printf("expected MMVQ launch: grid=(%lld,1,1) block=(32,%lld,1)\n",
                    static_cast<long long>(m / rows_per_block),
                    static_cast<long long>(warps_per_block));
    } else {
        std::printf("MMQ prediction: Q6_K; N=%lld -> J=%lld; M%%128=0 -> fallback=false\n",
                    static_cast<long long>(n), static_cast<long long>(tile_j));
        std::printf("expected MMQ launch if selected: grid=(%lld,1,1) block=(32,8,1)\n",
                    static_cast<long long>(efficiency >= 90 ? ntiles : GB10_SMS));
    }
    std::puts("allocations and enforced allowances:");
    print_bytes("GPU tensor buffer", s.tensor_buffer);
    print_bytes("CPU reference tensor buffer", s.tensor_buffer);
    print_bytes("host Q6_K staging", s.weight);
    print_bytes("host F32 input", s.input);
    print_bytes("host CPU + GPU outputs", s.host_outputs);
    print_bytes("initialization scratch allowance", s.init_scratch);
    print_bytes("CUDA Q8_1 MMQ workspace", s.q8_workspace);
    print_bytes("CUDA Stream-K fixup", s.stream_k_fixup);
    print_bytes("GGML metadata allowance", METADATA_ALLOWANCE);
    print_bytes("unclassified backend reserve", BACKEND_RESERVE);
    print_bytes("GUARDED TOTAL", s.guarded_total);
    print_bytes("HARD LIMIT", HARD_LIMIT);
    if (s.guarded_total >= HARD_LIMIT) {
        throw std::runtime_error("predicted allocation reaches the hard 2 GiB limit");
    }
}

uint32_t mix32(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    return x ^ (x >> 16);
}

void initialize_data(std::vector<uint8_t> & weights, std::vector<float> & input) {
    auto * blocks = reinterpret_cast<block_q6_K *>(weights.data());
    const size_t nblocks = weights.size() / sizeof(block_q6_K);
    for (size_t i = 0; i < nblocks; ++i) {
        block_q6_K & b = blocks[i];
        uint32_t state = mix32(static_cast<uint32_t>(i) ^ 0x4b325136U);
        for (uint8_t & v : b.ql) { state = mix32(state + 0x9e3779b9U); v = uint8_t(state); }
        for (uint8_t & v : b.qh) { state = mix32(state + 0x9e3779b9U); v = uint8_t(state); }
        for (int8_t & v : b.scales) {
            state = mix32(state + 0x9e3779b9U);
            v = static_cast<int8_t>(1 + (state % 7));
        }
        b.d = ggml_fp32_to_fp16(1.0f / 512.0f);
    }
    for (size_t i = 0; i < input.size(); ++i) {
        const int value = int(mix32(static_cast<uint32_t>(i) ^ 0x4b32494eU) % 2001U) - 1000;
        input[i] = float(value) / 1000.0f;
    }
}

struct RunResult {
    std::vector<float> output;
    double milliseconds = 0.0;
    double minimum_ms = 0.0;
    double maximum_ms = 0.0;
};

RunResult run_backend(ggml_backend_t backend, const std::vector<uint8_t> & weights,
                      const std::vector<float> & input, bool cpu, int64_t m, int64_t k, int64_t n,
                      bool fused_swiglu, int warmup, int iterations) {
    const size_t context_bytes = METADATA_ALLOWANCE / 2;
    ggml_init_params params = { context_bytes, nullptr, true };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) throw std::runtime_error("ggml_init failed");

    ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_Q6_K, k, m);
    ggml_tensor * w_gate = fused_swiglu ? ggml_new_tensor_2d(ctx, GGML_TYPE_Q6_K, k, m) : nullptr;
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, n);
    ggml_tensor * up = ggml_mul_mat(ctx, w, x);
    ggml_tensor * y = up;
    if (fused_swiglu) {
        ggml_tensor * gate = ggml_mul_mat(ctx, w_gate, x);
        y = ggml_swiglu_split(ctx, gate, up);
        ggml_set_name(w_gate, "synthetic_q6k_gate_weight");
        ggml_set_name(gate, "q6k_gate_projection");
        ggml_set_name(up, "q6k_up_projection");
    }
    ggml_set_name(w, "synthetic_q6k_weight");
    ggml_set_name(x, "deterministic_f32_input");
    ggml_set_name(y, "f32_output");
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 8, false);
    ggml_build_forward_expand(graph, y);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) {
        ggml_free(ctx);
        throw std::runtime_error("backend tensor-buffer allocation failed");
    }
    if (ggml_backend_buffer_get_size(buffer) >= HARD_LIMIT) {
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        throw std::runtime_error("backend tensor buffer violates the hard limit");
    }

    ggml_backend_tensor_set(w, weights.data(), 0, weights.size());
    if (w_gate) ggml_backend_tensor_set(w_gate, weights.data(), 0, weights.size());
    ggml_backend_tensor_set(x, input.data(), 0, input.size() * sizeof(float));
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
            throw std::runtime_error(std::string("warmup graph compute failed: ") + ggml_status_to_string(status));
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
            throw std::runtime_error(std::string("graph compute failed: ") + ggml_status_to_string(status));
        }
        samples.push_back(std::chrono::duration<double, std::milli>(stop - start).count());
    }
    std::sort(samples.begin(), samples.end());

    RunResult result;
    result.output.resize(m * n);
    ggml_backend_tensor_get(y, result.output.data(), 0, result.output.size() * sizeof(float));
    result.milliseconds = samples[samples.size() / 2];
    result.minimum_ms = samples.front();
    result.maximum_ms = samples.back();
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
        error2 += delta * delta;
        reference2 += double(reference[i]) * reference[i];
        max_abs = std::max(max_abs, std::abs(delta));
    }
    return reference2 == 0.0 ? double(error2) : double(error2 / reference2);
}

} // namespace

int main(int argc, char ** argv) try {
    constexpr const char * OUTPUT_PREFIX =
        "/home/dvijraicha/k2-gx10/results/q6k-prefill-j128-unroll4/outputs/";
    bool execute = false;
    bool fused_swiglu = false;
    bool allow_root_profile = false;
    std::string gpu_output_path;
    int64_t rows = DEFAULT_M;
    int64_t k = DEFAULT_K;
    int64_t columns = DEFAULT_N;
    int warmup = 0;
    int iterations = 1;
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "--execute") execute = true;
        else if (arg == "--dry-run") execute = false;
        else if (arg == "--fused-swiglu") fused_swiglu = true;
        else if (arg == "--allow-root-profile") allow_root_profile = true;
        else if (arg == "--rows" && i + 1 < argc) rows = std::stoll(argv[++i]);
        else if (arg == "--k" && i + 1 < argc) k = std::stoll(argv[++i]);
        else if (arg == "--columns" && i + 1 < argc) columns = std::stoll(argv[++i]);
        else if (arg == "--warmup" && i + 1 < argc) warmup = std::stoi(argv[++i]);
        else if (arg == "--iterations" && i + 1 < argc) iterations = std::stoi(argv[++i]);
        else if (arg == "--gpu-output" && i + 1 < argc) gpu_output_path = argv[++i];
        else throw std::runtime_error(
            "accepted options: --dry-run, --execute, --fused-swiglu, --rows M, --k K, --columns N, "
            "--warmup N, --iterations N, --gpu-output PATH, and --allow-root-profile; model paths are forbidden");
    }
    if (columns < 1 || columns > MAX_N) {
        throw std::runtime_error("--columns must be in 1.." + std::to_string(MAX_N));
    }
    if (fused_swiglu && columns != 1) {
        throw std::runtime_error("--fused-swiglu requires --columns 1 to match the decode fusion path");
    }
    if (rows < TILE_I || rows > MAX_DIM || rows % TILE_I != 0) {
        throw std::runtime_error("--rows must be in 128..28672 and divisible by 128");
    }
    if (k < QK || k > MAX_DIM || k % QK != 0) {
        throw std::runtime_error("--k must be in 256..28672 and divisible by 256");
    }
    if (warmup < 0 || warmup > 20) throw std::runtime_error("--warmup must be in 0..20");
    if (iterations < 1 || iterations > 100) {
        throw std::runtime_error("--iterations must be in 1..100");
    }
    if (geteuid() == 0 && !allow_root_profile) {
        throw std::runtime_error("refusing root execution without explicit --allow-root-profile");
    }
    if (geteuid() != 0 && allow_root_profile) {
        throw std::runtime_error("--allow-root-profile is reserved for the privileged profiler path");
    }
    if (!gpu_output_path.empty()) {
        const std::string prefix(OUTPUT_PREFIX);
        if (!execute || gpu_output_path.compare(0, prefix.size(), prefix) != 0 ||
                gpu_output_path.find("..") != std::string::npos) {
            throw std::runtime_error("--gpu-output is restricted to the experiment outputs directory");
        }
        if (access(gpu_output_path.c_str(), F_OK) == 0) {
            throw std::runtime_error("--gpu-output refuses to overwrite an existing path");
        }
    }
    print_plan(execute, fused_swiglu, rows, k, columns, warmup, iterations);
    if (!execute) {
        std::puts("DRY RUN PASS: no backend initialized and no GPU memory allocated");
        return 0;
    }

    const Sizes sizes(rows, k, columns, fused_swiglu);
    if (sizes.guarded_total >= HARD_LIMIT) throw std::runtime_error("hard allocation limit exceeded");
    std::vector<uint8_t> weights(sizes.weight);
    std::vector<float> input(k * columns);
    initialize_data(weights, input);

    ggml_backend_t cpu = ggml_backend_cpu_init();
    if (!cpu) throw std::runtime_error("CPU backend initialization failed");
    RunResult reference = run_backend(cpu, weights, input, true, rows, k, columns, fused_swiglu, 0, 1);
    ggml_backend_free(cpu);

    if (ggml_backend_cuda_get_device_count() < 1) throw std::runtime_error("no CUDA device found");
    ggml_backend_t cuda = ggml_backend_cuda_init(0);
    if (!cuda) throw std::runtime_error("CUDA backend initialization failed");
    RunResult actual = run_backend(cuda, weights, input, false, rows, k, columns, fused_swiglu, warmup, iterations);
    ggml_backend_free(cuda);

    if (!gpu_output_path.empty()) {
        std::FILE * output = std::fopen(gpu_output_path.c_str(), "wb");
        if (!output) throw std::runtime_error("failed to open --gpu-output path");
        const size_t written = std::fwrite(actual.output.data(), sizeof(float), actual.output.size(), output);
        const int close_status = std::fclose(output);
        if (written != actual.output.size() || close_status != 0) {
            throw std::runtime_error("failed to write complete GPU output");
        }
    }

    double max_abs = 0.0;
    const double error = nmse(reference.output, actual.output, max_abs);
    std::printf("CPU reference time: %.3f ms\n", reference.milliseconds);
    std::printf("CUDA timing: median=%.3f ms min=%.3f ms max=%.3f ms iterations=%d warmup=%d\n",
                actual.milliseconds, actual.minimum_ms, actual.maximum_ms, iterations, warmup);
    std::printf("correctness: NMSE=%.9g threshold=%.9g max_abs=%.9g\n", error, MAX_NMSE, max_abs);
    if (!(error <= MAX_NMSE)) throw std::runtime_error("CPU/CUDA correctness comparison failed");
    std::puts("EXECUTION PASS");
    return 0;
} catch (const std::exception & e) {
    std::fprintf(stderr, "ERROR: %s\n", e.what());
    return 1;
}
