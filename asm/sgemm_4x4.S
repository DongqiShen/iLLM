.text
; .align 4
.global _KERNEL_4X4_V2, _KERNEL_4X4

.macro _KERNEL_4X4_INNER ;4个a中的元素和4个b中的元素相乘，产生16个c中的元素
    // ARM64汇编代码
    // 寄存器 x0、x1、x2 分别存储着参数 a、b、c
    // 这里可以对参数进行操作或者执行其他指令

    ; 加载a_pack和b的第一行，4个元素，并将指针指向下一行
    ld1 {v0.4s}, [x0] ; v0 = a_pack
    add x0, x0, #16
    ld1 {v1.4s}, [x1] ; v1 = b
    add x1, x1, #16

    fmla v10.4s, v1.4s, v0.s[0] ;v10 = [a00b00, a00b01, a00b02, a00b03]
    fmla v11.4s, v1.4s, v0.s[1] ;v11 = [a10b00, a10b01, a10b02, a10b03]
    fmla v12.4s, v1.4s, v0.s[2] ;v12 = [a20b00, a20b01, a20b02, a20b03]
    fmla v13.4s, v1.4s, v0.s[3] ;v13 = [a30b00, a30b01, a30b02, a30b03]

.endm

.macro SAVE_4X4
    ld1 {v0.4s}, [x2]
    fadd v10.4s, v10.4s, v0.4s
    st1 {v10.4s}, [x2]

    add x13, x2, x3
    ld1 {v1.4s}, [x13]
    fadd v11.4s, v11.4s, v1.4s
    st1 {v11.4s}, [x13]

    add x13, x13, x3
    ld1 {v2.4s}, [x13]
    fadd v12.4s, v12.4s, v2.4s
    st1 {v12.4s}, [x13]

    add x13, x13, x3
    ld1 {v3.4s}, [x13]
    fadd v13.4s, v13.4s, v3.4s
    st1 {v13.4s}, [x13]
.endm

; 计算c中的一个4x4的矩阵，a中mc * k的一个panel和b的k * n的一个panel
_KERNEL_4X4:
    ; asr x4, x4, 2
    prfm pldl1keep, [x0, #64] ;从矩阵a中加载16个单精度浮点数
    prfm pldl1keep, [x1, #64] ;从矩阵b中加载16个单精度浮点数
    _KERNEL_4X4_INNER ;a += 16, b += 16
    _KERNEL_4X4_INNER ;a += 32, b += 32
    _KERNEL_4X4_INNER ;a += 48, b += 48
    _KERNEL_4X4_INNER ;a += 64, b += 64
    subs x4, x4, #1
    bne _KERNEL_4X4
    SAVE_4X4
    ret


_KERNEL_4X4_V2:
    ; 加载a_pack和b的第一行，4个元素，分别放在两个寄存器中，并将指针指向第二行
    ld1 {v0.2s, v1.2s}, [x0]  ; v0, v1 = a_pack
    add x0, x0, #16
    ld1 {v2.2s, v3.2s}, [x1]  ; v1, v2 = b
    add x1, x1, #16

    fmla v10.2s, v2.2s, v0.s[0] ;v10 = [a00b00, a00b01]
    fmla v11.2s, v2.2s, v0.s[0] ;v11 = [a00b02, a00b03]

    fmla v12.2s, v2.2s, v0.s[1] ;v12 = [a10b00, a10b01]
    fmla v13.2s, v2.2s, v0.s[1] ;v13 = [a10b02, a10b03]

    fmla v14.2s, v2.2s, v1.s[0] ;v14 = [a20b00, a20b01]
    fmla v15.2s, v2.2s, v1.s[0] ;v15 = [a20b02, a20b03]

    fmla v16.2s, v2.2s, v1.s[1] ;v16 = [a30b00, a30b01]
    fmla v17.2s, v2.2s, v1.s[1] ;v17 = [a30b02, a30b03]

    ; 加载a_pack和b的第二行，4个元素，分别放在两个寄存器中，并将指针指向第三行
    ld1 {v0.2s, v1.2s}, [x0]  ; v0, v1 = a_pack
    add x0, x0, #16
    ld1 {v2.2s, v3.2s}, [x1]  ; v1, v2 = b
    add x1, x1, #16

    fmla v10.2s, v2.2s, v0.s[0] ;v10 = [a01b10, a01b11]
    fmla v11.2s, v2.2s, v0.s[0] ;v11 = [a01b12, a01b13]

    fmla v12.2s, v2.2s, v0.s[1] ;v12 = [a11b10, a11b11]
    fmla v13.2s, v2.2s, v0.s[1] ;v13 = [a11b12, a11b13]

    fmla v14.2s, v2.2s, v1.s[0] ;v14 = [a21b10, a21b11]
    fmla v15.2s, v2.2s, v1.s[0] ;v15 = [a21b12, a21b13]

    fmla v16.2s, v2.2s, v1.s[1] ;v16 = [a31b10, a31b11]
    fmla v17.2s, v2.2s, v1.s[1] ;v17 = [a31b12, a31b13]

    ; 加载a_pack和b的第三行，4个元素，分别放在两个寄存器中，并将指针指向第四行
    ld1 {v0.2s, v1.2s}, [x0]  ; v0, v1 = a_pack
    add x0, x0, #16
    ld1 {v2.2s, v3.2s}, [x1]  ; v1, v2 = b
    add x1, x1, #16

    fmla v10.2s, v2.2s, v0.s[0] ;v10 = [a02b20, a02b21]
    fmla v11.2s, v2.2s, v0.s[0] ;v11 = [a02b22, a02b23]

    fmla v12.2s, v2.2s, v0.s[1] ;v12 = [a12b20, a12b21]
    fmla v13.2s, v2.2s, v0.s[1] ;v13 = [a12b22, a12b23]

    fmla v14.2s, v2.2s, v1.s[0] ;v14 = [a22b20, a22b21]
    fmla v15.2s, v2.2s, v1.s[0] ;v15 = [a22b22, a22b23]

    fmla v16.2s, v2.2s, v1.s[1] ;v16 = [a32b20, a32b21]
    fmla v17.2s, v2.2s, v1.s[1] ;v17 = [a32b22, a32b23]

    ; 加载a_pack和b的第四行
    ld1 {v0.2s, v1.2s}, [x0]  ; v0, v1 = a_pack
    ld1 {v2.2s, v3.2s}, [x1]  ; v1, v2 = b

    fmla v10.2s, v2.2s, v0.s[0] ;v10 = [a03b30, a03b31]
    fmla v11.2s, v2.2s, v0.s[0] ;v11 = [a03b32, a03b33]

    fmla v12.2s, v2.2s, v0.s[1] ;v12 = [a13b30, a13b31]
    fmla v13.2s, v2.2s, v0.s[1] ;v13 = [a13b32, a13b33]

    fmla v14.2s, v2.2s, v1.s[0] ;v14 = [a23b30, a23b31]
    fmla v15.2s, v2.2s, v1.s[0] ;v15 = [a23b32, a23b33]

    fmla v16.2s, v2.2s, v1.s[1] ;v16 = [a33b30, a33b31]
    fmla v17.2s, v2.2s, v1.s[1] ;v17 = [a33b32, a33b33]

    st1 {v10.2s, v11.2s, v12.2s, v13.2s}, [x2]
    add x2, x2, #32
    st1 {v14.2s, v15.2s, v16.2s, v17.2s}, [x2]

    ret
