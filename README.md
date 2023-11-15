# RECORD

## VSCODE DEBUG

在DEBUG CONSOLE中查看寄存器**v0**中的值：
```sh
po (float __attribute__((ext_vector_type(4)))) $v0
```

查看指针a中的16个元素：
```sh
a,[16]
```
