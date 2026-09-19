#pragma once

// llama.cpp's internal common type declarations are opt-in. Keep the opt-in
// local to this standalone benchmark rather than changing the source tree.
#define GGML_COMMON_DECL_CPP
#ifndef K2_GGML_COMMON_HEADER
#error "K2_GGML_COMMON_HEADER must name llama.cpp's ggml/src/ggml-common.h"
#endif
#include K2_GGML_COMMON_HEADER
