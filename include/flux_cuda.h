#ifndef FLUX_CUDA_H
#define FLUX_CUDA_H

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

// FLUX opcodes
#define FLUX_MOVI  0x2B
#define FLUX_IADD  0x08
#define FLUX_ISUB  0x09
#define FLUX_IMUL  0x0A
#define FLUX_IDIV  0x0B
#define FLUX_INC   0x0E
#define FLUX_DEC   0x0F
#define FLUX_CMP   0x2D
#define FLUX_JZ    0x05
#define FLUX_JNZ   0x06
#define FLUX_JMP   0x04
#define FLUX_HALT  0x80
#define FLUX_PUSH  0x20
#define FLUX_POP   0x21

// VM configuration
#define FLUX_NUM_REGS 16
#define FLUX_STACK_SIZE 256

// VM state for each thread
typedef struct {
    int32_t regs[FLUX_NUM_REGS];
    int32_t stack[FLUX_STACK_SIZE];
    int32_t sp;
    int32_t pc;
    uint8_t* bytecode;
    int32_t bytecode_len;
} FluxGPUVM;

// Host-side API
void flux_cuda_init(int num_vms);
void flux_cuda_load(const uint8_t* bytecode, int bytecode_len);
void flux_cuda_run(int num_vms, int32_t* inputs);
void flux_cuda_read_results(int32_t* results);
void flux_cuda_free();

#ifdef __cplusplus
}
#endif

#endif // FLUX_CUDA_H
