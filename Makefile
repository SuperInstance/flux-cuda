CC = gcc
NVCC = nvcc
CFLAGS = -Wall -Wextra -O2
NVCCFLAGS = -O2 -std=c++11
INCLUDES = -I./include
BUILD_DIR = build
BIN_DIR = bin

TARGET = $(BIN_DIR)/flux_cuda_demo

all: check_cuda $(TARGET)

check_cuda:
	@which $(NVCC) >/dev/null 2>&1 || (echo "CUDA not found, building CPU-only version"; $(MAKE) cpu_only)

cpu_only: $(BIN_DIR)/flux_cpu_demo

$(BIN_DIR)/flux_cpu_demo: src/main.cu
	@mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) $(INCLUDES) -DCPU_ONLY -o $@ $< -lm

$(TARGET): src/flux_cuda.cu src/main.cu
	@mkdir -p $(BUILD_DIR) $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $@ $^

clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR)

.PHONY: all clean check_cuda cpu_only
