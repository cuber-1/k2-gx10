# Compile and static resource record

Status: static-gate rejection; no GPU kernel was executed.

## Provenance

- GPU target: NVIDIA GB10, compute capability 12.1; CMake resolved both builds
  to `sm_121a` with CUDA 13.0.88.
- Reused fresh baseline executable SHA256:
  `4e62ca4b25b80494cab49ced1dedea5eb25c539d2cd21285e6d05ba7e87735a6`.
- Reused fresh baseline CUDA library SHA256:
  `87bece1ef046ea5a5efd14ea1865aeb18ae2086135e6bb6bc38bac9137ca395a`.
- Frozen harness SHA256:
  `1378536c577aac6d8d31d0cd63e2b676480052982dac4d6274c0e32923627bbb`.
- Candidate source differs from accepted source only in
  `ggml/src/ggml-cuda/CMakeLists.txt`, applying
  `-Xptxas=--register-usage-level=4` to the absolute Q6_K MMQ source path.

Generated and verbose executed commands confirm Q6_K alone receives the flag.
Path-normalized Q5_K and Q8_0 commands are byte-identical to baseline, and the
flag is absent from the CUDA-library link command.

## PTX and SASS

The raw embedded PTX differs in one `$str$1` global containing
`NO_DEVICE_CODE(__FILE__)`: the isolated source-tree path and its byte-array
length differ. Extraction metadata also records the candidate ptxas option.
Both raw artifacts and their diff are preserved. After removing that single
path-only global line, the complete ordered PTX directives/instructions are
byte-identical with SHA256
`0b90e6f82eaad979250ab467dce2908ae934cca5329bad5d3fb07ee9f327aaed`.

The full exact-Q6 cubin disassemblies are byte-identical with SHA256
`19726e72e7f164dc12f322b73e42a43228b70264afde7f7498be6ce94596b7fe`.
An independent per-function comparison extracted ordered 128-bit instruction
encodings (opcode plus control word): all 64/64 Q6 functions are identical.
The raw cubin metadata differs only because the embedded source path is five
bytes shorter, reflected by Common GLOBAL 27,180 to 27,175 bytes.

## Resource screen

| Function/metric | Baseline | Level 4 |
|---|---:|---:|
| J128,false registers | 255 | 255 |
| J128,false stack | 64 B | 64 B |
| J128,false shared | 1024 B | 1024 B |
| J128,false static LDL | 32 | 32 |
| J128,false static STL | 27 | 27 |
| J128,false static IMMA | 512 | 512 |
| J32,false registers/stack/shared | 254 / 0 B / 1024 B | identical |
| J128,false fixup registers/stack/shared | 120 / 0 B / 1536 B | identical |
| J32,false fixup registers/stack/shared | 46 / 0 B / 1152 B | identical |

Because the target SASS is identical, its terminal full-tile loop and all
loop-carried spill behavior are also identical. There is no qualifying stack
or hot-local-traffic improvement.
