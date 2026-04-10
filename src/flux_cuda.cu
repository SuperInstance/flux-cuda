#include "flux_cuda.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

// Device pointers
static uint8_t* d_bytecode = nullptr;
static int32_t* d_inputs = nullptr;
static int32_t* d_results = nullptr;
static int32_t* d_regs = nullptr;
static int d_bytecode_len = 0;
static int d_num_vms = 0;

// Device function to execute one instruction
__device__ void flux_cuda_execute(FluxGPUVM* vm) {
    while (1) {
        if (vm->pc >= vm->bytecode_len) break;
        
        uint8_t opcode = vm->bytecode[vm->pc];
        vm->pc++;
        
        switch (opcode) {
            case FLUX_MOVI: {
                uint8_t reg = vm->bytecode[vm->pc];
                vm->pc++;
                int32_t imm = (vm->bytecode[vm->pc]) |
                             (vm->bytecode[vm->pc + 1] << 8) |
                             (vm->bytecode[vm->pc + 2] << 16) |
                             (vm->bytecode[vm->pc + 3] << 24);
                vm->pc += 4;
                vm->regs[reg] = imm;
                break;
            }
            case FLUX_IADD: {
                uint8_t rd = vm->bytecode[vm->pc];
                vm->pc++;
                uint8_t rs = vm->bytecode[vm->pc];
                vm->pc++;
                vm->regs[rd] += vm->regs[rs];
                break;
            }
            case FLUX_ISUB: {
                uint8_t rd = vm->bytecode[vm->pc];
                vm->pc++;
                uint8_t rs = vm->bytecode[vm->pc];
                vm->pc++;
                vm->regs[rd] -= vm->regs[rs];
                break;
            }
            case FLUX_IMUL: {
                uint8_t rd = vm->bytecode[vm->pc];
                vm->pc++;
                uint8_t rs = vm->bytecode[vm->pc];
                vm->pc++;
                vm->regs[rd] *= vm->regs[rs];
                break;
            }
            case FLUX_IDIV: {
                uint8_t rd = vm->bytecode[vm->pc];
                vm->pc++;
                uint8_t rs = vm->bytecode[vm->pc];
                vm->pc++;
                if (vm->regs[rs] != 0) {
                    vm->regs[rd] /= vm->regs[rs];
                }
                break;
            }
            case FLUX_INC: {
                uint8_t reg = vm->bytecode[vm->pc];
                vm->pc++;
                vm->regs[reg]++;
                break;
            }
            case FLUX_DEC: {
                uint8_t reg = vm->bytecode[vm->pc];
                vm->pc++;
                vm->regs[reg]--;
                break;
            }
            case FLUX_CMP: {
                uint8_t r1 = vm->bytecode[vm->pc];
                vm->pc++;
                uint8_t r2 = vm->bytecode[vm->pc];
                vm->pc++;
                // For simplicity, store comparison result in register 0
                vm->regs[0] = (vm->regs[r1] - vm->regs[r2]);
                break;
            }
            case FLUX_JZ: {
                uint8_t reg = vm->bytecode[vm->pc];
                vm->pc++;
                int32_t offset = (vm->bytecode[vm->pc]) |
                                (vm->bytecode[vm->pc + 1] << 8) |
                                (vm->bytecode[vm->pc + 2] << 16) |
                                (vm->bytecode[vm->pc + 3] << 24);
                vm->pc += 4;
                if (vm->regs[reg] == 0) {
                    vm->pc += offset;
                }
                break;
            }
            case FLUX_JNZ: {
                uint8_t reg = vm->bytecode[vm->pc];
                vm->pc++;
                int32_t offset = (vm->bytecode[vm->pc]) |
                                (vm->bytecode[vm->pc + 1] << 8) |
                                (vm->bytecode[vm->pc + 2] << 16) |
                                (vm->bytecode[vm->pc + 3] << 24);
                vm->pc += 4;
                if (vm->regs[reg] != 0) {
                    vm->pc += offset;
                }
                break;
            }
            case FLUX_JMP: {
                int32_t offset = (vm->bytecode[vm->pc]) |
                                (vm->bytecode[vm->pc + 1] << 8) |
                                (vm->bytecode[vm->pc + 2] << 16) |
                                (vm->bytecode[vm->pc + 3] << 24);
                vm->pc += 4;
                vm->pc += offset;
                break;
            }
            case FLUX_PUSH: {
                uint8_t reg = vm->bytecode[vm->pc];
                vm->pc++;
                if (vm->sp < FLUX_STACK_SIZE) {
                    vm->stack[vm->sp] = vm->regs[reg];
                    vm->sp++;
                }
                break;
            }
            case FLUX_POP: {
                uint8_t reg = vm->bytecode[vm->pc];
                vm->pc++;
                if (vm->sp > 0) {
                    vm->sp--;
                    vm->regs[reg] = vm->stack[vm->sp];
                }
                break;
            }
            case FLUX_HALT:
                return;
            default:
                // Unknown opcode
                return;
        }
    }
}

// Kernel to run multiple VM instances
__global__ void flux_cuda_batch(uint8_t* bytecode, int bytecode_len, 
                                int32_t* inputs, int32_t* results, int num_vms) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_vms) return;
    
    // Each thread gets its own VM state
    FluxGPUVM vm;
    // Initialize VM state to zero
    for (int i = 0; i < FLUX_NUM_REGS; i++) {
        vm.regs[i] = 0;
    }
    for (int i = 0; i < FLUX_STACK_SIZE; i++) {
        vm.stack[i] = 0;
    }
    vm.sp = 0;
    vm.pc = 0;
    vm.bytecode = bytecode;
    vm.bytecode_len = bytecode_len;
    
    // Initialize with input
    vm.regs[0] = inputs[idx];  // Input in register 0
    
    // Execute
    flux_cuda_execute(&vm);
    
    // Store result (register 0 holds the result)
    results[idx] = vm.regs[0];
}

// Host-side implementation
void flux_cuda_init(int num_vms) {
    d_num_vms = num_vms;
    cudaMalloc(&d_inputs, num_vms * sizeof(int32_t));
    cudaMalloc(&d_results, num_vms * sizeof(int32_t));
    cudaMalloc(&d_regs, num_vms * FLUX_NUM_REGS * sizeof(int32_t));
}

void flux_cuda_load(const uint8_t* bytecode, int bytecode_len) {
    d_bytecode_len = bytecode_len;
    cudaMalloc(&d_bytecode, bytecode_len);
    cudaMemcpy(d_bytecode, bytecode, bytecode_len, cudaMemcpyHostToDevice);
}

void flux_cuda_run(int num_vms, int32_t* inputs) {
    // Copy inputs to device
    cudaMemcpy(d_inputs, inputs, num_vms * sizeof(int32_t), cudaMemcpyHostToDevice);
    
    // Launch kernel
    int blockSize = 256;
    int gridSize = (num_vms + blockSize - 1) / blockSize;
    flux_cuda_batch<<<gridSize, blockSize>>>(d_bytecode, d_bytecode_len, 
                                             d_inputs, d_results, num_vms);
    cudaDeviceSynchronize();
}

void flux_cuda_read_results(int32_t* results) {
    cudaMemcpy(results, d_results, d_num_vms * sizeof(int32_t), cudaMemcpyDeviceToHost);
}

void flux_cuda_free() {
    if (d_bytecode) cudaFree(d_bytecode);
    if (d_inputs) cudaFree(d_inputs);
    if (d_results) cudaFree(d_results);
    if (d_regs) cudaFree(d_regs);
    d_bytecode = nullptr;
    d_inputs = nullptr;
    d_results = nullptr;
    d_regs = nullptr;
}
