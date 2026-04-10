#include "flux_cuda.h"
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cstdint>

// CPU implementation for comparison
int32_t factorial_cpu(int32_t n) {
    int32_t result = 1;
    for (int32_t i = 2; i <= n; i++) {
        result *= i;
    }
    return result;
}

// Generate bytecode for factorial:
// Input in r0, output in r0
// Algorithm: result = 1, i = 1
// while i <= n: result *= i; i++
// We'll implement using a counter that counts down from n to 0
void generate_factorial_bytecode(uint8_t* buffer, int* length) {
    int pos = 0;
    
    // r1 = 1 (result)
    buffer[pos++] = FLUX_MOVI;
    buffer[pos++] = 1;
    buffer[pos++] = 1; buffer[pos++] = 0; buffer[pos++] = 0; buffer[pos++] = 0;
    
    // r2 = 1 (i)
    buffer[pos++] = FLUX_MOVI;
    buffer[pos++] = 2;
    buffer[pos++] = 1; buffer[pos++] = 0; buffer[pos++] = 0; buffer[pos++] = 0;
    
    // loop:
    // Compare r2 and r0+1
    // We need to check if i > n, which is when i - (n+1) >= 0
    // Let's compute r3 = r0 + 1
    // Copy r0 to r3 using addition with zero
    // First, set r3 to 0
    buffer[pos++] = FLUX_MOVI;
    buffer[pos++] = 3;
    buffer[pos++] = 0; buffer[pos++] = 0; buffer[pos++] = 0; buffer[pos++] = 0;
    // r3 = r0 + r3 (which is just r0)
    buffer[pos++] = FLUX_IADD;
    buffer[pos++] = 3;
    buffer[pos++] = 0;
    // Add 1 to r3
    buffer[pos++] = FLUX_MOVI;
    buffer[pos++] = 4;
    buffer[pos++] = 1; buffer[pos++] = 0; buffer[pos++] = 0; buffer[pos++] = 0;
    buffer[pos++] = FLUX_IADD;
    buffer[pos++] = 3;
    buffer[pos++] = 4;
    
    // Now compare r2 and r3
    buffer[pos++] = FLUX_CMP;
    buffer[pos++] = 2;
    buffer[pos++] = 3;
    
    // If r2 >= r3, jump to end (offset needs to be calculated)
    // We'll use JNZ on the comparison result in r0
    // The offset will be set later, for now use placeholder (0)
    int jmp_pos = pos;
    buffer[pos++] = FLUX_JNZ;
    buffer[pos++] = 0;  // Register 0 holds comparison result
    buffer[pos++] = 0; buffer[pos++] = 0; buffer[pos++] = 0; buffer[pos++] = 0;
    
    // Multiply result by i: r1 = r1 * r2
    buffer[pos++] = FLUX_IMUL;
    buffer[pos++] = 1;
    buffer[pos++] = 2;
    
    // Increment i: r2++
    buffer[pos++] = FLUX_INC;
    buffer[pos++] = 2;
    
    // Jump back to loop start
    buffer[pos++] = FLUX_JMP;
    // Calculate offset: jump back to the start of loop (right after initializing r2)
    // The loop starts at instruction index 12 (after both MOVI instructions)
    // Current position is pos, we want to jump to instruction at index 12
    // Offset = 12 - pos - 5 (because JMP instruction takes 5 bytes)
    int offset = 12 - pos - 5;
    buffer[pos++] = (offset >> 0) & 0xFF;
    buffer[pos++] = (offset >> 8) & 0xFF;
    buffer[pos++] = (offset >> 16) & 0xFF;
    buffer[pos++] = (offset >> 24) & 0xFF;
    
    // end: move result to r0
    int end_pos = pos;
    buffer[pos++] = FLUX_MOVI;
    buffer[pos++] = 0;
    // Copy r1 to r0: we need to use addition since there's no direct move
    // First, set r0 to 0
    buffer[pos++] = 0; buffer[pos++] = 0; buffer[pos++] = 0; buffer[pos++] = 0;
    // Then add r1 to r0
    buffer[pos++] = FLUX_IADD;
    buffer[pos++] = 0;
    buffer[pos++] = 1;
    
    // Halt
    buffer[pos++] = FLUX_HALT;
    
    // Now fix the jump offset at jmp_pos
    // We want to jump to end_pos when condition is true
    // The offset is end_pos - (jmp_pos + 5)
    int jmp_offset = end_pos - (jmp_pos + 5);
    buffer[jmp_pos + 2] = (jmp_offset >> 0) & 0xFF;
    buffer[jmp_pos + 3] = (jmp_offset >> 8) & 0xFF;
    buffer[jmp_pos + 4] = (jmp_offset >> 16) & 0xFF;
    buffer[jmp_pos + 5] = (jmp_offset >> 24) & 0xFF;
    
    *length = pos;
}

int main() {
    const int NUM_VMS = 1000;
    int32_t inputs[NUM_VMS];
    int32_t gpu_results[NUM_VMS];
    int32_t cpu_results[NUM_VMS];
    
    // Initialize inputs
    for (int i = 0; i < NUM_VMS; i++) {
        inputs[i] = (i % 10) + 1;  // Numbers from 1 to 10
    }
    
    // Compute CPU results
    clock_t cpu_start = clock();
    for (int i = 0; i < NUM_VMS; i++) {
        cpu_results[i] = factorial_cpu(inputs[i]);
    }
    clock_t cpu_end = clock();
    double cpu_time = (double)(cpu_end - cpu_start) / CLOCKS_PER_SEC;
    
    // Generate factorial bytecode
    uint8_t bytecode[256];
    int bytecode_len;
    generate_factorial_bytecode(bytecode, &bytecode_len);
    
    // Initialize GPU
    flux_cuda_init(NUM_VMS);
    flux_cuda_load(bytecode, bytecode_len);
    
    // Run on GPU
    clock_t gpu_start = clock();
    flux_cuda_run(NUM_VMS, inputs);
    clock_t gpu_end = clock();
    double gpu_time = (double)(gpu_end - gpu_start) / CLOCKS_PER_SEC;
    
    // Read results
    flux_cuda_read_results(gpu_results);
    
    // Verify results
    int errors = 0;
    for (int i = 0; i < NUM_VMS; i++) {
        if (gpu_results[i] != cpu_results[i]) {
            errors++;
            if (errors < 5) {
                printf("Mismatch at %d: GPU=%d, CPU=%d\n", i, gpu_results[i], cpu_results[i]);
            }
        }
    }
    
    printf("FLUX CUDA VM Demo\n");
    printf("Number of VMs: %d\n", NUM_VMS);
    printf("CPU time: %.6f seconds\n", cpu_time);
    printf("GPU time: %.6f seconds\n", gpu_time);
    printf("Speedup: %.2fx\n", cpu_time / gpu_time);
    printf("Errors: %d\n", errors);
    
    // Cleanup
    flux_cuda_free();
    return 0;
}
