# Result: blocked before GPU

The requested Q8_0-K/F16-V experiment cannot use Flash Attention with the
frozen accepted FA_ALL_QUANTS=OFF library. Static dispatch returns no CUDA
Flash Attention kernel for the mixed pair, violating the mandatory no-fallback
gate by construction.

No harness code, model load, GPU work, or timing was performed. A separately
authorized ALL_QUANTS=ON build would be a new controlled experiment and must
include OFF-vs-ON F16 parity before testing Q8_0 K cache.
