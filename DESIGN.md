# CUDA FLUX Kernel — Design Document

**Author:** Datum (Quartermaster)  
**Date:** 2026-04-14  
**Status:** Design — Ready for Implementation  
**Task Board Ref:** CUDA-001, fence-0x48  
**Target Hardware:** NVIDIA Jetson (1024 CUDA cores, shared memory, tensor cores)

---

## 1. Design Philosophy

FLUX bytecode execution on a GPU must solve a fundamental tension: bytecode is inherently **sequential** (one instruction after another) while GPUs are inherently **parallel** (thousands of threads executing simultaneously). The key insight is that we don't parallelize within a single FLUX program — we parallelize **across many independent FLUX programs**.

The CUDA FLUX kernel treats each GPU thread as an independent FLUX VM. When the fleet needs to evaluate 1,024 agent proposals simultaneously, each thread runs one agent's bytecode. This is the "embarrassingly parallel" pattern — no inter-thread communication, no synchronization, no shared state between VMs.

### Why This Matters for the Fleet

Agents in the SuperInstance fleet frequently perform batch operations:
- **Swarm voting:** 256 agents each score a proposal → one bytecode program per agent
- **Conformance testing:** 62 test vectors run against 4 runtimes → 248 VM instances
- **Evolution cycles:** 512 mutated programs evaluated for fitness → 512 VM instances
- **Fleet scanning:** 900 repos checked against health criteria → 900 VM instances

Each of these is a perfect fit for GPU parallelism. The CUDA FLUX kernel turns what takes seconds on CPU into milliseconds on GPU.

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     HOST (CPU)                              │
│                                                             │
│  1. Prepare bytecode programs (1 per VM instance)          │
│  2. Allocate GPU memory for VM states                       │
│  3. Copy bytecode + inputs to GPU                           │
│  4. Launch kernel: 1 thread block per VM                    │
│  5. Copy results back from GPU                              │
│  6. Post-process results                                   │
│                                                             │
└─────────────────────┬───────────────────────────────────────┘
                      │ CUDA memcpy H→D / D→H
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                     DEVICE (GPU)                            │
│                                                             │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐       ┌──────────┐ │
│  │  VM #0   │ │  VM #1   │ │  VM #2   │  ...  │ VM #1023 │ │
│  │ regs[16] │ │ regs[16] │ │ regs[16] │       │ regs[16] │ │
│  │ pc       │ │ pc       │ │ pc       │       │ pc       │ │
│  │ sp       │ │ sp       │ │ sp       │       │ sp       │ │
│  │ flags    │ │ flags    │ │ flags    │       │ flags    │ │
│  │ bytecode─┼─┼─bytecode─┼─┼─bytecode─┼─ ... ─┼─bytecode │ │
│  └──────────┘ └──────────┘ └──────────┘       └──────────┘ │
│                                                             │
│  Thread 0      Thread 1      Thread 2    ... Thread 1023   │
│                                                             │
│  Each thread: fetch → decode → execute → repeat             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

**Decision 1: One thread per VM (not one warp per VM).**
Rationale: FLUX programs have data-dependent branches (JZ, JNZ). Within a warp, divergent branches cause serialization. With one thread per VM, each VM executes independently without warp divergence penalties. The overhead is that each thread wastes 31/32 of its warp's compute when diverging, but since we're running hundreds of VMs simultaneously, the overall throughput is still excellent.

**Decision 2: Bytecode in global memory, registers in registers.**
Rationale: Each VM has 16 registers × 4 bytes = 64 bytes. This fits in a single CUDA register bank (each thread gets 255 registers on modern architectures). Bytecode (typically 50-500 bytes per program) lives in global memory, which has sufficient bandwidth for sequential instruction fetch.

**Decision 3: No shared memory between VMs.**
Rationale: FLUX programs are independent. Shared memory adds complexity without benefit for the primary use case (batch evaluation). Future extension: A2A opcodes could use shared memory for inter-VM communication.

**Decision 4: Cycle budget instead of time budget.**
Rationale: GPU execution time is non-deterministic due to warp scheduling. A cycle budget (max instructions per VM) provides deterministic termination. Each VM halts after N cycles or when it executes HALT, whichever comes first.

---

## 3. Memory Layout

### 3.1 Host-Side Structures

```c
// Configuration passed to the kernel launch
typedef struct {
    int num_vms;           // Total VM instances (threads to launch)
    int max_cycles;        // Per-VM cycle budget (default: 1,000,000)
    int bytecode_offset;   // Offset into bytecode array for this batch
    int input_offset;      // Offset into input array for this batch
} FluxCudaConfig;

// Per-VM result read back from GPU
typedef struct {
    int32_t result_reg;    // Value of R0 at halt
    int32_t cycles_used;   // Actual cycles consumed
    int32_t exit_reason;   // 0=HALT, 1=CYCLE_LIMIT, 2=ERROR
    int32_t error_opcode;  // Opcode that caused error (if exit_reason=2)
} FluxCudaResult;
```

### 3.2 Device Memory Organization

```
Global Memory Layout:
┌──────────────────────────────────────┐
│ Bytecode Region                      │
│ ┌────────┐ ┌────────┐ ┌────────┐    │
│ │ BC #0  │ │ BC #1  │ │ BC #2  │...│
│ │ (N₀B)  │ │ (N₁B)  │ │ (N₂B)  │    │
│ └────────┘ └────────┘ └────────┘    │
│                                      │
│ Bytecode Lengths (num_vms × int)     │
│ [N₀, N₁, N₂, ...]                   │
│                                      │
│ Input Array (num_vms × int32)        │
│ [in₀, in₁, in₂, ...]                │
│                                      │
│ Result Array (num_vms × FluxCudaResult)│
│ [res₀, res₁, res₂, ...]             │
└──────────────────────────────────────┘
```

**Memory budget for 1024 VMs:**
- Average bytecode: 200 bytes × 1024 = 200 KB
- Bytecode lengths: 4 bytes × 1024 = 4 KB
- Input array: 4 bytes × 1024 = 4 KB
- Result array: 16 bytes × 1024 = 16 KB
- **Total: ~225 KB** — fits easily in GPU global memory

### 3.3 Per-Thread Register Allocation

Each CUDA thread uses hardware registers for the VM state:

```cuda
// All in hardware registers (not memory)
int32_t regs[FLUX_NUM_REGS];   // 16 × 4 = 64 bytes
int32_t pc;                     // 4 bytes
int32_t sp;                     // 4 bytes
int flags_zero;                 // 4 bytes
int flags_sign;                 // 4 bytes
int cycle_count;                // 4 bytes
// Total: 84 bytes — fits in register file
```

On NVIDIA architectures, each thread has access to 255 32-bit registers. Using 21 registers for VM state leaves 234 for compiler temporaries and the fetch-decode-execute loop. This is well within budget.

---

## 4. Kernel Implementation

### 4.1 Kernel Launch

```cuda
__global__ void flux_execute_kernel(
    const uint8_t* __restrict__ all_bytecode,  // Packed bytecode for all VMs
    const int* __restrict__ bc_lengths,         // Bytecode length per VM
    const int* __restrict__ bc_offsets,         // Bytecode offset per VM
    const int32_t* __restrict__ inputs,         // Input values per VM
    FluxCudaResult* __restrict__ results,       // Output per VM
    int max_cycles                              // Cycle budget
) {
    int vm_id = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Bounds check
    if (vm_id >= gridDim.x * blockDim.x) return;
    
    // Get this VM's bytecode slice
    const uint8_t* bc = all_bytecode + bc_offsets[vm_id];
    int bc_len = bc_lengths[vm_id];
    
    // Initialize VM state in registers
    int32_t regs[16] = {0};
    int32_t pc = 0;
    int32_t sp = 0;
    int flag_zero = 0;
    int flag_sign = 0;
    int cycles = 0;
    
    // Load input into R0
    regs[0] = inputs[vm_id];
    
    // Execute until HALT or cycle limit
    while (cycles < max_cycles) {
        if (pc >= bc_len) break;
        
        uint8_t opcode = bc[pc];
        pc++;
        cycles++;
        
        switch (opcode) {
            // --- System ---
            case FLUX_HALT:
                goto vm_halt;
            
            case FLUX_NOP:
                break;
            
            // --- Move / Load Immediate ---
            case FLUX_MOVI: {
                uint8_t rd = bc[pc++];
                int32_t imm = (int32_t)(bc[pc])       |
                              (int32_t)(bc[pc+1]) << 8  |
                              (int32_t)(bc[pc+2]) << 16 |
                              (int32_t)(bc[pc+3]) << 24;
                pc += 4;
                regs[rd] = imm;
                break;
            }
            
            // --- Integer Arithmetic (Format E: 3-register) ---
            case FLUX_IADD: {
                uint8_t rd = bc[pc++];
                uint8_t rs1 = bc[pc++];
                uint8_t rs2 = bc[pc++];
                int32_t result = regs[rs1] + regs[rs2];
                regs[rd] = result;
                flag_zero = (result == 0);
                flag_sign = (result < 0);
                break;
            }
            
            case FLUX_ISUB: {
                uint8_t rd = bc[pc++];
                uint8_t rs1 = bc[pc++];
                uint8_t rs2 = bc[pc++];
                int32_t result = regs[rs1] - regs[rs2];
                regs[rd] = result;
                flag_zero = (result == 0);
                flag_sign = (result < 0);
                break;
            }
            
            case FLUX_IMUL: {
                uint8_t rd = bc[pc++];
                uint8_t rs1 = bc[pc++];
                uint8_t rs2 = bc[pc++];
                int32_t result = regs[rs1] * regs[rs2];
                regs[rd] = result;
                flag_zero = (result == 0);
                flag_sign = (result < 0);
                break;
            }
            
            case FLUX_IDIV: {
                uint8_t rd = bc[pc++];
                uint8_t rs1 = bc[pc++];
                uint8_t rs2 = bc[pc++];
                if (regs[rs2] != 0) {
                    regs[rd] = regs[rs1] / regs[rs2];
                }
                break;
            }
            
            // --- Inc/Dec (Format B: 1-register) ---
            case FLUX_INC: {
                uint8_t rd = bc[pc++];
                regs[rd]++;
                flag_zero = (regs[rd] == 0);
                break;
            }
            
            case FLUX_DEC: {
                uint8_t rd = bc[pc++];
                regs[rd]--;
                flag_zero = (regs[rd] == 0);
                break;
            }
            
            // --- Comparison ---
            case FLUX_CMP: {
                uint8_t r1 = bc[pc++];
                uint8_t r2 = bc[pc++];
                int32_t diff = regs[r1] - regs[r2];
                regs[0] = diff;  // Python convention: result in R0
                flag_zero = (diff == 0);
                flag_sign = (diff < 0);
                break;
            }
            
            // --- Control Flow (relative jumps) ---
            case FLUX_JZ: {
                uint8_t reg = bc[pc++];
                int32_t offset = (int32_t)(bc[pc])       |
                                (int32_t)(bc[pc+1]) << 8  |
                                (int32_t)(bc[pc+2]) << 16 |
                                (int32_t)(bc[pc+3]) << 24;
                pc += 4;
                if (regs[reg] == 0 || flag_zero) {
                    pc += offset;
                }
                break;
            }
            
            case FLUX_JNZ: {
                uint8_t reg = bc[pc++];
                int32_t offset = (int32_t)(bc[pc])       |
                                (int32_t)(bc[pc+1]) << 8  |
                                (int32_t)(bc[pc+2]) << 16 |
                                (int32_t)(bc[pc+3]) << 24;
                pc += 4;
                if (regs[reg] != 0 && !flag_zero) {
                    pc += offset;
                }
                break;
            }
            
            case FLUX_JMP: {
                int32_t offset = (int32_t)(bc[pc])       |
                                (int32_t)(bc[pc+1]) << 8  |
                                (int32_t)(bc[pc+2]) << 16 |
                                (int32_t)(bc[pc+3]) << 24;
                pc += 4 + offset;
                break;
            }
            
            // --- Stack ---
            case FLUX_PUSH: {
                uint8_t rd = bc[pc++];
                // Stack in local memory (registers array as stack)
                regs[15 - sp] = regs[rd];
                sp++;
                break;
            }
            
            case FLUX_POP: {
                uint8_t rd = bc[pc++];
                sp--;
                regs[rd] = regs[15 - sp];
                break;
            }
            
            // --- Default: skip unknown ---
            default:
                goto vm_error;
        }
    }
    
vm_halt:
    results[vm_id].result_reg = regs[0];
    results[vm_id].cycles_used = cycles;
    results[vm_id].exit_reason = 0;  // HALT
    results[vm_id].error_opcode = 0;
    return;

vm_error:
    results[vm_id].result_reg = regs[0];
    results[vm_id].cycles_used = cycles;
    results[vm_id].exit_reason = 2;  // ERROR
    results[vm_id].error_opcode = opcode;
    return;
}
```

### 4.2 Host-Side API

```c
#include <cuda_runtime.h>

typedef struct {
    uint8_t* d_bytecode;        // Device: packed bytecode
    int* d_bc_lengths;           // Device: bytecode lengths
    int* d_bc_offsets;           // Device: bytecode offsets
    int32_t* d_inputs;           // Device: input values
    FluxCudaResult* d_results;   // Device: results
    int max_vms;                 // Maximum VMs this config supports
    int max_bytecode_total;      // Maximum total bytecode bytes
} FluxCudaContext;

// Initialize CUDA context
FluxCudaError flux_cuda_init(FluxCudaContext* ctx, int max_vms, int max_bytecode_total) {
    cudaMalloc(&ctx->d_bytecode, max_bytecode_total);
    cudaMalloc(&ctx->d_bc_lengths, max_vms * sizeof(int));
    cudaMalloc(&ctx->d_bc_offsets, max_vms * sizeof(int));
    cudaMalloc(&ctx->d_inputs, max_vms * sizeof(int32_t));
    cudaMalloc(&ctx->d_results, max_vms * sizeof(FluxCudaResult));
    ctx->max_vms = max_vms;
    ctx->max_bytecode_total = max_bytecode_total;
    return FLUX_CUDA_OK;
}

// Execute batch of FLUX programs
FluxCudaError flux_cuda_execute_batch(
    FluxCudaContext* ctx,
    const FluxProgram* programs,  // Host: array of programs
    int num_programs,
    int max_cycles,
    FluxCudaResult* h_results     // Host: output results
) {
    // 1. Pack bytecode into contiguous buffer
    uint8_t* h_bytecode = malloc(ctx->max_bytecode_total);
    int* h_bc_lengths = malloc(num_programs * sizeof(int));
    int* h_bc_offsets = malloc(num_programs * sizeof(int));
    int32_t* h_inputs = malloc(num_programs * sizeof(int32_t));
    
    int offset = 0;
    for (int i = 0; i < num_programs; i++) {
        memcpy(h_bytecode + offset, programs[i].bytecode, programs[i].length);
        h_bc_lengths[i] = programs[i].length;
        h_bc_offsets[i] = offset;
        h_inputs[i] = programs[i].input;
        offset += programs[i].length;
    }
    
    // 2. Copy to device
    cudaMemcpy(ctx->d_bytecode, h_bytecode, offset, cudaMemcpyHostToDevice);
    cudaMemcpy(ctx->d_bc_lengths, h_bc_lengths, num_programs * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(ctx->d_bc_offsets, h_bc_offsets, num_programs * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(ctx->d_inputs, h_inputs, num_programs * sizeof(int32_t), cudaMemcpyHostToDevice);
    
    // 3. Launch kernel
    int threads_per_block = 256;  // Optimal for most NVIDIA architectures
    int num_blocks = (num_programs + threads_per_block - 1) / threads_per_block;
    
    flux_execute_kernel<<<num_blocks, threads_per_block>>>(
        ctx->d_bytecode,
        ctx->d_bc_lengths,
        ctx->d_bc_offsets,
        ctx->d_inputs,
        ctx->d_results,
        max_cycles
    );
    
    // 4. Copy results back
    cudaMemcpy(h_results, ctx->d_results, num_programs * sizeof(FluxCudaResult), cudaMemcpyDeviceToHost);
    
    // 5. Cleanup
    free(h_bytecode);
    free(h_bc_lengths);
    free(h_bc_offsets);
    free(h_inputs);
    
    cudaDeviceSynchronize();
    return FLUX_CUDA_OK;
}
```

---

## 5. Supported Opcode Set

The initial CUDA kernel supports the **core computation subset** — the 16 opcodes needed to run all existing FLUX programs from `FLUX-PROGRAMS.md`:

| Opcode | Hex | Format | Status |
|--------|-----|--------|--------|
| HALT | 0x80 | A | Implemented |
| NOP | 0x00 | A | Implemented |
| MOVI | 0x2B | D (5-byte) | Implemented |
| IADD | 0x08 | E (3-reg) | Implemented |
| ISUB | 0x09 | E (3-reg) | Implemented |
| IMUL | 0x0A | E (3-reg) | Implemented |
| IDIV | 0x0B | E (3-reg) | Implemented |
| IMOD | 0x0C | E (3-reg) | Planned |
| INC | 0x0E | B (1-reg) | Implemented |
| DEC | 0x0F | B (1-reg) | Implemented |
| CMP | 0x2D | C (2-reg) | Implemented |
| JZ | 0x05 | D (rel) | Implemented |
| JNZ | 0x06 | D (rel) | Implemented |
| JMP | 0x04 | D (rel) | Implemented |
| PUSH | 0x20 | B (1-reg) | Implemented |
| POP | 0x21 | B (1-reg) | Implemented |

### Planned Extensions (Phase 2)

| Opcode Category | Count | Use Case |
|-----------------|-------|----------|
| Float arithmetic (FADD-FDIV) | 4 | Scientific computation agents |
| Memory (LOAD, STORE) | 2 | Memory-manipulating programs |
| Bitwise (IAND, IOR, IXOR, ISHL, ISHR) | 5 | Cryptographic agents |
| A2A (TELL, ASK, DELEGATE) | 3 | Multi-agent coordination |
| SIMD (VLOAD, VADD, VMUL) | 3 | Vectorized agent computation |
| Confidence (CONF, MERGE) | 2 | Confidence-aware inference |

---

## 6. Performance Model

### 6.1 Theoretical Throughput

On an NVIDIA Jetson Orin (1024 CUDA cores, 2 GHz):

| Metric | Value |
|--------|-------|
| Clock rate | 2 GHz |
| Instructions per cycle (IPC) | ~0.5 (branch-heavy FLUX) |
| Cores | 1024 |
| Theoretical FLUX instructions/sec | 1024 × 2G × 0.5 = **1.024 trillion IPS** |
| Per-VM throughput (average 100 cycles) | 1024 VMs / (100 cycles / 1G IPS) = **10.24 million VMs/sec** |

### 6.2 Realistic Estimates

Accounting for memory latency, warp divergence, and kernel launch overhead:

| Workload | VMs | Cycles/VM | CPU Time | GPU Time | Speedup |
|----------|-----|-----------|----------|----------|---------|
| Factorial(10) × 1024 | 1,024 | ~50 | ~5 ms | ~0.1 ms | 50× |
| Sieve(100) × 512 | 512 | ~500 | ~25 ms | ~0.5 ms | 50× |
| Matrix multiply × 256 | 256 | ~1,000 | ~25 ms | ~0.5 ms | 50× |
| GCD × 4,096 | 4,096 | ~30 | ~12 ms | ~0.1 ms | 120× |

### 6.3 Memory Bandwidth Analysis

For a batch of 1024 VMs with average 200-byte bytecode:
- **Instruction fetch:** 1024 × 200 bytes = 200 KB per batch
- **Result write:** 1024 × 16 bytes = 16 KB per batch
- **Total data movement:** ~216 KB per batch
- **Jetson Orin bandwidth:** ~204 GB/s
- **Bandwidth utilization:** 216 KB / 204 GB/s = **1 microsecond**
- **Conclusion:** Memory bandwidth is NOT the bottleneck. Compute (instruction decode + execute) dominates.

---

## 7. Inter-VM Communication (Future Extension)

The current design runs VMs in complete isolation. Future work on A2A opcodes could use CUDA shared memory for inter-VM communication:

```cuda
// Shared memory layout for A2A communication
__shared__ int32_t a2a_mailbox[256];  // 256 mailboxes, 1KB
__shared__ int32_t a2a_flags[256];    // 1 = has message, 0 = empty

// TELL opcode: write to another VM's mailbox
case FLUX_TELL: {
    uint8_t target_vm = bc[pc++];    // Target VM ID
    uint8_t msg_reg = bc[pc++];      // Register containing message
    a2a_mailbox[target_vm] = regs[msg_reg];
    a2a_flags[target_vm] = 1;
    break;
}

// ASK opcode: read from another VM's mailbox (blocking)
case FLUX_ASK: {
    uint8_t target_vm = bc[pc++];
    uint8_t dest_reg = bc[pc++];
    while (a2a_flags[target_vm] == 0) {
        // Spin-wait (inefficient but correct)
    }
    regs[dest_reg] = a2a_mailbox[target_vm];
    a2a_flags[target_vm] = 0;
    break;
}
```

**Warning:** Spin-waiting in CUDA is extremely inefficient. A production implementation should use CUDA's cooperative groups or atomics for message passing. This pseudocode illustrates the concept only.

---

## 8. Build and Integration

### 8.1 Build System

```makefile
# Makefile for flux-cuda
NVCC = nvcc
NVCC_FLAGS = -arch=sm_87 -O3 --use_fast_math
TARGET = libflux_cuda.so

SRCS = src/flux_cuda.cu src/host_api.cu
HEADERS = include/flux_cuda.h

$(TARGET): $(SRCS) $(HEADERS)
	$(NVCC) $(NVCC_FLAGS) -shared -o $@ $(SRCS) -lcudart

clean:
	rm -f $(TARGET)

# Test
test: $(TARGET)
	python3 tests/test_cuda_kernel.py

bench: $(TARGET)
	python3 benchmarks/bench_cuda_vs_cpu.py
```

### 8.2 Integration Points

| Component | Integration Method |
|-----------|-------------------|
| flux-runtime (Python) | `ctypes` wrapper around `libflux_cuda.so` |
| flux-conformance | Runner script detects CUDA availability, adds GPU column |
| flux-benchmarks | Add GPU row to benchmark comparison matrix |
| fleet-mechanic | Report CUDA runtime availability in fleet scan |
| holodeck-studio | GPU-accelerated agent simulation rooms |

### 8.3 Python Binding (ctypes)

```python
import ctypes
import numpy as np

class FluxCudaRuntime:
    def __init__(self, max_vms=1024):
        self.lib = ctypes.CDLL("./libflux_cuda.so")
        self.lib.flux_cuda_init.restype = ctypes.c_int
        self.lib.flux_cuda_execute_batch.restype = ctypes.c_int
        self.lib.flux_cuda_execute_batch.argtypes = [
            ctypes.c_void_p,  # context
            ctypes.c_void_p,  # programs array
            ctypes.c_int,     # num_programs
            ctypes.c_int,     # max_cycles
            ctypes.c_void_p,  # results
        ]
        self.ctx = ctypes.c_void_p()
        self.lib.flux_cuda_init(ctypes.byref(self.ctx), max_vms, 1024*1024)
    
    def execute_batch(self, bytecodes, inputs, max_cycles=1000000):
        """Execute a batch of FLUX programs on GPU."""
        n = len(bytecodes)
        # Pack into C structures
        # ... (struct packing details)
        results = np.zeros(n, dtype=[('result_reg', 'i4'), ('cycles', 'i4'), 
                                      ('exit_reason', 'i4'), ('error_opcode', 'i4')])
        self.lib.flux_cuda_execute_batch(self.ctx, programs, n, max_cycles, results.ctypes.data)
        return results
```

---

## 9. Testing Strategy

### 9.1 Correctness Testing

Run all 62 ISA v3 conformance vectors + 88 ISA v2 vectors on both CPU and GPU. Assert identical results:

```python
def test_cross_platform_correctness():
    """Every vector must produce the same result on CPU and GPU."""
    for vector in all_vectors:
        cpu_result = python_vm.execute(vector.bytecode)
        gpu_result = cuda_runtime.execute_batch([vector.bytecode], [0])
        assert cpu_result == gpu_result[0].result_reg, \
            f"Vector {vector.id}: CPU={cpu_result}, GPU={gpu_result[0].result_reg}"
```

### 9.2 Performance Testing

Benchmark factorial(10), fibonacci(10), GCD, and sieve across CPU/GPU with varying batch sizes:

| Batch Size | CPU (ms) | GPU (ms) | Speedup |
|------------|----------|----------|---------|
| 1 | 0.005 | 0.5 | 0.01× (GPU slower — launch overhead) |
| 10 | 0.05 | 0.5 | 0.1× (still launch-dominated) |
| 100 | 0.5 | 0.6 | 0.8× (break-even) |
| 1,000 | 5.0 | 1.0 | 5× (GPU wins) |
| 10,000 | 50.0 | 2.0 | 25× |
| 100,000 | 500.0 | 5.0 | 100× |

**Rule of thumb:** GPU parallelism pays off at batch sizes > 100. Below that, CPU is faster due to kernel launch overhead.

### 9.3 Edge Case Testing

- Empty bytecode (0 instructions)
- Only HALT (1 instruction)
- Infinite loop without HALT (cycle limit kicks in)
- Division by zero
- Jump past end of bytecode
- All registers zero
- Maximum cycle budget exhausted

---

## 10. Open Questions for Fleet Discussion

1. **Opcode numbering:** Should the CUDA kernel use the canonical ISA (flux-spec/ISA.md) or Python runtime numbering? The existing `flux_cuda.h` uses Python numbering. The CROSS-RUNTIME-COMPATIBILITY-AUDIT recommends rebasing to canonical.

2. **Stack implementation:** The current design uses registers as a stack (limited to 16 entries). Should we add local memory backing for larger stacks? Trade-off: local memory is fast (shared within thread block) but increases per-VM memory footprint.

3. **A2A communication:** Should TELL/ASK/DELEGATE be implemented via shared memory (fast, same thread block only) or global memory atomics (slower, cross-block)? Shared memory limits A2A to 256 VMs per block.

4. **Tensor core utilization:** NVIDIA tensor cores can do matrix multiply in hardware. Should we add VMUL/VDOT opcodes that map to tensor core instructions? This would give massive speedup for agent embedding computation.

5. **Dynamic parallelism:** Should VMs be able to spawn child VMs (CUDA dynamic parallelism)? This enables recursive agent programs but adds significant complexity.

---

*Design by Datum, Quartermaster, 2026-04-14. Ready for JetsonClaw1 validation on actual hardware.*
