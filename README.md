# flux-cuda

**FLUX CUDA Runtime** — GPU-accelerated bytecode VM for parallel agent execution.

Each GPU thread runs its own FLUX VM instance. Launch thousands of agents simultaneously.

## Architecture

```
Host                    GPU
─────                   ────
Load bytecode ───────►  Global Memory (shared, read-only)
                         │
Create N inputs ──────►  Input Array (per-thread)
                         │
Launch kernel ─────────► ┌──────────────────────────┐
                         │ Thread 0: VM(R0=3)  → F! │
                         │ Thread 1: VM(R0=4)  → F! │
                         │ Thread 2: VM(R0=5)  → F! │
                         │ ...                       │
                         │ Thread N: VM(R0=N) → F!  │
                         └──────────────────────────┘
                                    │
Read results ◄──────────  Output Array (per-thread R0)
```

## Building

```bash
# With CUDA
make

# Without CUDA (CPU fallback)
make cpu
```

## Example

```c
// Launch 1000 factorial computations on GPU
int N = 1000;
int* inputs = malloc(N * sizeof(int));
int* outputs = malloc(N * sizeof(int));
for (int i = 0; i < N; i++) inputs[i] = i;

FluxCUDAContext ctx;
flux_cuda_init(&ctx, factorial_bytecode, bc_len);
flux_cuda_run(&ctx, inputs, outputs, N);
flux_cuda_free(&ctx);

// outputs[i] now contains factorial(inputs[i])
```

## License

MIT — SuperInstance (DiGennaro et al.)
