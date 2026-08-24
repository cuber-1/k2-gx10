# Result: terminal Q8-index recurrence CSE

Decision: **rejected at the static gate**.

The targeted cc1210 terminal-path index sharing is algebraically correct and
reduces the J128/nonfallback main symbol by eight instructions, while leaving
IMMA, FFMA, global/shared operations, barriers, and branch counts unchanged.
It nevertheless worsens the exact resource this experiment was intended to
improve: stack grows from 64 to 80 bytes, LDL from 32 to 34, and STL from 27 to
29. The predeclared GO gate required stack <=64, LDL <=30, STL <=26, and fewer
terminal sum spills.

Because spill traffic worsened, the candidate was rejected before completing a
full library build. No correctness execution, performance timing, Nsight
Systems, Nsight Compute, model load, or full-model run occurred.

The source change is isolated to `vendor/llama-prefill-q8-index-cse`; accepted
source/build artifacts and `/home/dvijraicha/llama.cpp` are untouched. Preserve
this as a rejected indexing formulation so the same compiler shape is not
repeated without new register-allocation evidence.
