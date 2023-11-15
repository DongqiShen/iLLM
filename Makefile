# 编译器设置
CC := clang
AS := as
CFLAGS := -Wall -Wextra -g
LDFLAGS :=

# 源文件目录
SRC_DIR := src
ASM_DIR := asm

# 源文件扩展名
SRC_EXT := c
ASM_EXT := S

# 生成文件目录
BUILD_DIR := build

# 获取所有的源文件和汇编文件
SRC := $(wildcard $(SRC_DIR)/*.$(SRC_EXT))
ASM := $(wildcard $(ASM_DIR)/*.$(ASM_EXT))

# 将源文件转换为目标文件
OBJ := $(patsubst $(SRC_DIR)/%.$(SRC_EXT),$(BUILD_DIR)/%.o,$(SRC))
OBJ += $(patsubst $(ASM_DIR)/%.$(ASM_EXT),$(BUILD_DIR)/%.o,$(ASM))

# 目标可执行文件
TARGET := $(BUILD_DIR)/main

# 默认目标
all: $(TARGET)

# 生成可执行文件
$(TARGET): $(OBJ)
	$(CC) $(LDFLAGS) $^ -o $@

# 生成目标文件
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.$(SRC_EXT)
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: $(ASM_DIR)/%.$(ASM_EXT)
	@mkdir -p $(@D)
	$(AS) $(CFLAGS) $< -o $@

# 清理目标文件和可执行文件
clean:
	rm -rf $(BUILD_DIR) $(TARGET)

# 防止make在文件夹中生成名为clean的文件
.PHONY: clean
