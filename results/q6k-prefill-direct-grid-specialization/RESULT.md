# Result: rejected before GPU

The exact cc1210 Q6_K/J128/nonfallback direct-grid specialization successfully
removed the unreachable terminal/fixup math body: the emitted direct=true
kernel contains one 256-IMMA/640-FFMA body, preserves 1,024 B shared memory, and
does not access the inherited unused `tmp_fixup` ABI parameter.

It did not eliminate compiler spills. The direct kernel still uses 40 B stack,
15 LDL, and 11 STL at 255 registers, versus the required zero stack/local
traffic. This is a mandatory static rejection, so no correctness run, timing,
Nsight profiling, or full-model test was performed.

Isolation evidence is clean: after explicitly mapping baseline three-template
main names to candidate four-template `<...,false>` names, all 32 generic main
functions and all 32 Stream-K fixup functions are exact ordered-encoding and
resource matches. The accepted source/build remains untouched.
