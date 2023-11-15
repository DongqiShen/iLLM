# iLLm

从零开始用C实现大语言模型的推理，计划首先支持Mac m1，并以Qwen-int4版本为例。

## VSCODE DEBUG

在DEBUG CONSOLE中查看寄存器**v0**中的值：
```sh
po (float __attribute__((ext_vector_type(4)))) $v0
```

查看指针a中的16个元素：
```sh
a,[16]
```
