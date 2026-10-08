// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later
//
// ARM64 instruction emitter — generates ARM64 machine code for the
// x86-64 → ARM64 JIT backend. Emits into a growable buffer; the caller
// is responsible for making the buffer executable (via mmap + mprotect)
// before jumping to the emitted code.
//
// Only the instructions needed for the JIT's hot-path translation are
// implemented: MOV (reg/mem/imm), ADD, SUB, AND, OR, XOR, CMP, branches,
// load/store, and the exit-to-host trampoline. The rest fall back to
// the interpreter.
//
// Register allocation (fixed, no spilling needed for the MVP):
//   x86-64 RAX..R15  →  ARM64 X0..X15   (16 GPRs, 1:1 mapping)
//   x86-64 RIP        →  ARM64 X16       (link-like scratch)
//   x86-64 RFLAGS     →  ARM64 X17       (raw RFLAGS word)
//   x86-64 RSP        →  ARM64 X4        (already mapped as RBX→X3, RSP→X4)
//
// Wait — Zydis order is RAX=0 RCX=1 RDX=2 RBX=3 RSP=4 RBP=5 RSI=6 RDI=7
// R8=8..R15=15. We map index 0→X0, 1→X1, ..., 15→X15. RSP is index 4 → X4.
// RIP is separate (X16), RFLAGS is separate (X17).
//
// ARM64 calling convention: X0-X7 args, X30=LR, SP=X31/separate.
// We keep X18 as a "state pointer" (points to X64CpuState on stack/heap).
// X19-X28 are callee-saved; we don't use them in the JIT for simplicity.

#pragma once

#include <cstdint>
#include <cstring>
#include <vector>

#include "common/types.h"

namespace Core::Cpu::Jit {

// ARM64 register names matching the hardware encoding (0-31).
enum Arm64Reg : u32 {
    X0 = 0, X1, X2, X3, X4, X5, X6, X7,
    X8, X9, X10, X11, X12, X13, X14, X15,
    X16, X17, X18, X19, X20, X21, X22, X23,
    X24, X25, X26, X27, X28, X29, X30, X31 = 31,
    // X31 as SP or ZR depending on context
    SP = 31,
    ZR = 31,
};

// ARM64 condition codes (inverted from x86-64, but we map them here).
enum class Cond : u32 {
    EQ = 0x0, NE = 0x1, CS = 0x2, CC = 0x3,
    MI = 0x4, PL = 0x5, VS = 0x6, VC = 0x7,
    HI = 0x8, LS = 0x9, GE = 0xA, LT = 0xB,
    GT = 0xC, LE = 0xD, AL = 0xE, NV = 0xF,
};

// Code emission buffer. Grows as instructions are emitted. The caller
// must flush_dcache / make executable before jumping to the code.
class CodeBuffer {
public:
    CodeBuffer() = default;
    explicit CodeBuffer(size_t initial_capacity) {
        m_code.reserve(initial_capacity);
    }

    // Emit a single 32-bit ARM64 instruction.
    void Emit32(u32 insn) {
        // ARM64 is little-endian; write 4 bytes in LE order.
        m_code.push_back(insn & 0xFF);
        m_code.push_back((insn >> 8) & 0xFF);
        m_code.push_back((insn >> 16) & 0xFF);
        m_code.push_back((insn >> 24) & 0xFF);
    }

    // Get a pointer to the emitted code (not executable until made so).
    const u8* data() const { return m_code.data(); }
    size_t size() const { return m_code.size(); }
    size_t size_bytes() const { return m_code.size(); }

    // Clear the buffer (for recompiling a block).
    void clear() { m_code.clear(); }

    // Align to a 4-byte boundary (ARM64 instructions are always 4 bytes).
    void align4() {
        while (m_code.size() % 4 != 0) {
            m_code.push_back(0);
        }
    }

    // Emit a NOP (instruction encoding 0xD503201F).
    void Nop() { Emit32(0xD503201F); }

    // ── Register-register MOV ──
    // MOV Xd, Xs  →  ORR Xd, XZR, Xs
    void Mov(Arm64Reg dst, Arm64Reg src) {
        // ORR Xd, XZR, Xs  →  0xAA000000 | (src << 16) | (ZR << 5) | dst
        Emit32(0xAA000000 | (u32(src) << 16) | (u32(ZR) << 5) | u32(dst));
    }

    // ── Register-immediate MOV ──
    // MOVZ Xd, #imm16 (shift=0)  →  0xD2800000 | (imm16 << 5) | dst
    void Movz16(Arm64Reg dst, u16 imm, u32 shift = 0) {
        Emit32(0xD2800000 | ((shift / 16) << 21) | (u32(imm) << 5) | u32(dst));
    }

    // MOVK Xd, #imm16 (shift=N*16) — keep and insert
    void Movk16(Arm64Reg dst, u16 imm, u32 shift = 0) {
        Emit32(0xF2800000 | ((shift / 16) << 21) | (u32(imm) << 5) | u32(dst));
    }

    // MOV Xd, #imm64 (emits up to 4 MOVZ/MOVK instructions)
    void MovImm64(Arm64Reg dst, u64 imm) {
        // If the value fits in 16 bits, use a single MOVZ.
        if (imm <= 0xFFFF) {
            Movz16(dst, (u16)imm);
            return;
        }
        // Full 64-bit (or >16-bit): MOVZ + up to 3 MOVK
        Movz16(dst, (u16)(imm & 0xFFFF), 0);
        if (imm >> 16) Movk16(dst, (u16)((imm >> 16) & 0xFFFF), 16);
        if (imm >> 32) Movk16(dst, (u16)((imm >> 32) & 0xFFFF), 32);
        if (imm >> 48) Movk16(dst, (u16)((imm >> 48) & 0xFFFF), 48);
    }

    // ── ADD (register) ──
    // ADD Xd, Xn, Xm  →  0x8B000000 | (Xm << 16) | (Xn << 5) | Xd
    void Add(Arm64Reg dst, Arm64Reg lhs, Arm64Reg rhs) {
        Emit32(0x8B000000 | (u32(rhs) << 16) | (u32(lhs) << 5) | u32(dst));
    }

    // ── SUB (register) ──
    // SUB Xd, Xn, Xm  →  0xCB000000 | (Xm << 16) | (Xn << 5) | Xd
    void Sub(Arm64Reg dst, Arm64Reg lhs, Arm64Reg rhs) {
        Emit32(0xCB000000 | (u32(rhs) << 16) | (u32(lhs) << 5) | u32(dst));
    }

    // ── AND (register) ──
    // AND Xd, Xn, Xm  →  0x8A000000 | (Xm << 16) | (Xn << 5) | Xd
    void And(Arm64Reg dst, Arm64Reg lhs, Arm64Reg rhs) {
        Emit32(0x8A000000 | (u32(rhs) << 16) | (u32(lhs) << 5) | u32(dst));
    }

    // ── ORR (register) ──
    // ORR Xd, Xn, Xm  →  0xAA000000 | (Xm << 16) | (Xn << 5) | Xd
    void Orr(Arm64Reg dst, Arm64Reg lhs, Arm64Reg rhs) {
        Emit32(0xAA000000 | (u32(rhs) << 16) | (u32(lhs) << 5) | u32(dst));
    }

    // ── EOR (XOR, register) ──
    // EOR Xd, Xn, Xm  →  0xCA000000 | (Xm << 16) | (Xn << 5) | Xd
    void Eor(Arm64Reg dst, Arm64Reg lhs, Arm64Reg rhs) {
        Emit32(0xCA000000 | (u32(rhs) << 16) | (u32(lhs) << 5) | u32(dst));
    }

    // ── SUBS (sets flags, for CMP) ──
    // SUBS Xd, Xn, Xm  →  0xEB000000 | (Xm << 16) | (Xn << 5) | Xd
    void Subs(Arm64Reg dst, Arm64Reg lhs, Arm64Reg rhs) {
        Emit32(0xEB000000 | (u32(rhs) << 16) | (u32(lhs) << 5) | u32(dst));
    }

    // CMP Xn, Xm  →  SUBS XZR, Xn, Xm
    void Cmp(Arm64Reg lhs, Arm64Reg rhs) {
        Subs(ZR, lhs, rhs);
    }

    // ── Load (LDR) ──
    // LDR Xt, [Xn, #offset]  →  0xF9400000 | (offset/8 << 10) | (Xn << 5) | Xt
    // 64-bit load, scaled by 8, unsigned offset.
    void Ldr64(Arm64Reg dst, Arm64Reg base, u32 offset) {
        Emit32(0xF9400000 | ((offset / 8) << 10) | (u32(base) << 5) | u32(dst));
    }

    // ── Store (STR) ──
    // STR Xt, [Xn, #offset]  →  0xF9000000 | (offset/8 << 10) | (Xn << 5) | Xt
    void Str64(Arm64Reg src, Arm64Reg base, u32 offset) {
        Emit32(0xF9000000 | ((offset / 8) << 10) | (u32(base) << 5) | u32(src));
    }

    // ── Load 8-bit (LDRB) ──
    // LDRB Wt, [Xn, #offset]  →  0x39400000 | (offset << 10) | (Xn << 5) | Xt
    void Ldrb32(Arm64Reg dst, Arm64Reg base, u32 offset) {
        Emit32(0x39400000 | (offset << 10) | (u32(base) << 5) | u32(dst));
    }

    // ── Store 8-bit (STRB) ──
    // STRB Wt, [Xn, #offset]  →  0x39000000 | (offset << 10) | (Xn << 5) | Xt
    void Strb32(Arm64Reg src, Arm64Reg base, u32 offset) {
        Emit32(0x39000000 | (offset << 10) | (u32(base) << 5) | u32(src));
    }

    // ── Load 32-bit (LDR 32-bit) ──
    // LDR Wt, [Xn, #offset]  →  0xB9400000 | ((offset/4) << 10) | (Xn << 5) | Wt
    void Ldr32(Arm64Reg dst, Arm64Reg base, u32 offset) {
        Emit32(0xB9400000 | ((offset / 4) << 10) | (u32(base) << 5) | u32(dst));
    }

    // ── Store 32-bit (STR 32-bit) ──
    // STR Wt, [Xn, #offset]  →  0xB9000000 | ((offset/4) << 10) | (Xn << 5) | Wt
    void Str32(Arm64Reg src, Arm64Reg base, u32 offset) {
        Emit32(0xB9000000 | ((offset / 4) << 10) | (u32(base) << 5) | u32(src));
    }

    // ── Unconditional branch (immediate) ──
    // B #offset  →  0x14000000 | (offset/4 & 0x03FFFFFF)
    // offset is in bytes from this instruction.
    void B(s32 offset) {
        u32 imm26 = (u32)((offset / 4) & 0x03FFFFFF);
        Emit32(0x14000000 | imm26);
    }

    // ── Conditional branch (immediate) ──
    // B.cond #offset  →  0x54000000 | ((offset/4 & 0x7FFFF) << 5) | cond
    void Bcond(Cond cond, s32 offset) {
        u32 imm19 = (u32)((offset / 4) & 0x7FFFF);
        Emit32(0x54000000 | (imm19 << 5) | u32(cond));
    }

    // ── Branch with link (BL) ──
    // BL #offset  →  0x94000000 | (offset/4 & 0x03FFFFFF)
    void Bl(s32 offset) {
        u32 imm26 = (u32)((offset / 4) & 0x03FFFFFF);
        Emit32(0x94000000 | imm26);
    }

    // ── Return (RET) ──
    // RET Xn  →  0xD65F0000 | (Xn << 5)
    void Ret(Arm64Reg reg = X30) {
        Emit32(0xD65F0000 | (u32(reg) << 5));
    }

    // ── ADD (immediate, 12-bit) ──
    // ADD Xd, Xn, #imm12  →  0x91000000 | (imm12 << 10) | (Xn << 5) | Xd
    void AddImm(Arm64Reg dst, Arm64Reg src, u32 imm12) {
        Emit32(0x91000000 | ((imm12 & 0xFFF) << 10) | (u32(src) << 5) | u32(dst));
    }

    // ── SUB (immediate, 12-bit) ──
    // SUB Xd, Xn, #imm12  →  0xD1000000 | (imm12 << 10) | (Xn << 5) | Xd
    void SubImm(Arm64Reg dst, Arm64Reg src, u32 imm12) {
        Emit32(0xD1000000 | ((imm12 & 0xFFF) << 10) | (u32(src) << 5) | u32(dst));
    }

    // ── STR Xt, [Xn], #imm  (post-index) ──
    // Used for PUSH: STR Xt, [SP], #-16  (but SP alignment matters)
    void Str64PostIndex(Arm64Reg src, Arm64Reg base, s32 imm) {
        Emit32(0xF8000800 | ((u32(imm) & 0x1FF) << 12) | (u32(base) << 5) | u32(src));
    }

    // ── LDR Xt, [Xn], #imm  (post-index) ──
    // Used for POP: LDR Xt, [SP], #16
    void Ldr64PostIndex(Arm64Reg dst, Arm64Reg base, s32 imm) {
        Emit32(0xF8400400 | ((u32(imm) & 0x1FF) << 12) | (u32(base) << 5) | u32(dst));
    }

    // ── STR Xt, [Xn, #-imm]!  (pre-index) ──
    // Used for PUSH: STR Xt, [SP, #-16]!
    void Str64PreIndex(Arm64Reg src, Arm64Reg base, s32 imm) {
        Emit32(0xF8000C00 | ((u32(-imm) & 0x1FF) << 12) | (u32(base) << 5) | u32(src));
    }

    // ── LDR Xt, [Xn, #imm]!  (pre-index) ──
    // Used for POP: LDR Xt, [SP, #16]!
    void Ldr64PreIndex(Arm64Reg dst, Arm64Reg base, s32 imm) {
        Emit32(0xF8400C00 | ((u32(imm) & 0x1FF) << 12) | (u32(base) << 5) | u32(dst));
    }

    // ── Store pair (STP) ──
    // STP Xt1, Xt2, [Xn, #offset]
    void Stp64(Arm64Reg src1, Arm64Reg src2, Arm64Reg base, s32 offset) {
        u32 imm7 = (u32)((offset / 8) & 0x7F);
        Emit32(0xA9000000 | (imm7 << 15) | (u32(src2) << 10) | (u32(base) << 5) | u32(src1));
    }

    // ── Load pair (LDP) ──
    // LDP Xt1, Xt2, [Xn, #offset]
    void Ldp64(Arm64Reg dst1, Arm64Reg dst2, Arm64Reg base, s32 offset) {
        u32 imm7 = (u32)((offset / 8) & 0x7F);
        Emit32(0xA9400000 | (imm7 << 15) | (u32(dst2) << 10) | (u32(base) << 5) | u32(dst1));
    }

    // ── BR Xn (branch to register) ──
    // BR Xn  →  0xD61F0000 | (Xn << 5)
    void Br(Arm64Reg reg) {
        Emit32(0xD61F0000 | (u32(reg) << 5));
    }

    // ── BLR Xn (branch with link to register) ──
    // BLR Xn  →  0xD63F0000 | (Xn << 5)
    void Blr(Arm64Reg reg) {
        Emit32(0xD63F0000 | (u32(reg) << 5));
    }

    // ── ADR Xd, #offset (PC-relative address) ──
    // Computes the address of the current instruction + offset and stores it
    // in Xd. Used by the trampoline to set X30 (LR) to the exit-handler address
    // without needing a literal pool.
    // Encoding: bit 31 = 0 (ADR, not ADRP), bits 30-29 = immlo (low 2 bits),
    //           bits 28-24 = 10000, bits 23-5 = immhi (high 19 bits), bits 4-0 = Rd.
    void Adr(Arm64Reg dst, s32 offset) {
        u32 immlo = (u32)(offset & 0x3);
        u32 immhi = (u32)((offset >> 2) & 0x7FFFF);
        Emit32(0x10000000 | (immlo << 29) | (immhi << 5) | u32(dst));
    }

    // ── ANDS Xd, Xn, Xm (AND setting flags) ──
    // Same as AND but updates NZCV (N and Z set from result, C=0, V=0).
    // Used for TEST (when dest=ZR) and for flag-setting AND.
    // Encoding: 0xEA000000 | (Xm << 16) | (Xn << 5) | Xd
    void Ands(Arm64Reg dst, Arm64Reg lhs, Arm64Reg rhs) {
        Emit32(0xEA000000 | (u32(rhs) << 16) | (u32(lhs) << 5) | u32(dst));
    }

    // ── CBNZ Xn, #offset (compare and branch if non-zero, 64-bit) ──
    // Branches to PC + offset if Xn != 0. Used for Jcc condition testing.
    // Encoding: 0xB5000000 | (imm19 << 5) | Xn
    void Cbnz(Arm64Reg reg, s32 offset) {
        u32 imm19 = (u32)((offset / 4) & 0x7FFFF);
        Emit32(0xB5000000 | (imm19 << 5) | u32(reg));
    }

    // ── CBZ Xn, #offset (compare and branch if zero, 64-bit) ──
    // Branches to PC + offset if Xn == 0. Used for inverted Jcc conditions.
    // Encoding: 0xB4000000 | (imm19 << 5) | Xn
    void Cbz(Arm64Reg reg, s32 offset) {
        u32 imm19 = (u32)((offset / 4) & 0x7FFFF);
        Emit32(0xB4000000 | (imm19 << 5) | u32(reg));
    }

    // ── TST Xn, #imm12 (test bits, simplified immediate) ──
    // Alias for ANDS XZR, Xn, #imm. The imm12 is placed in the imms field with
    // N=0 and immr=0; this produces a valid ARM64 instruction (the actual
    // encoded immediate value is determined by the N:immr:imms bitfield
    // pattern, so callers should use values that match a real logical-immediate
    // pattern when correctness is required).
    // Encoding: 0xF2000000 | (imm12 << 10) | (Xn << 5) | ZR
    void TstImm(Arm64Reg reg, u32 imm12) {
        Emit32(0xF2000000 | ((imm12 & 0xFFF) << 10) | (u32(reg) << 5) | u32(ZR));
    }

    // ── MRS Xd, NZCV (read NZCV flags into a register) ──
    // Reads the ARM64 NZCV condition flags into bits 31-28 of Xd.
    // Used after flag-setting instructions to convert ARM64 NZCV to x86 RFLAGS.
    // Encoding: 0xD53B4200 | Rd
    void MrsNzcv(Arm64Reg dst) {
        Emit32(0xD53B4200 | u32(dst));
    }

    // ── LSL Xd, Xn, #shift (logical shift left immediate) ──
    // Alias for UBFM Xd, Xn, #(-shift mod 64), #(63 - shift).
    // Encoding (UBFM 64-bit): 0xD3400000 | (immr << 16) | (imms << 10) | (Xn << 5) | Xd
    void LslImm(Arm64Reg dst, Arm64Reg src, u32 shift) {
        u32 immr = (64 - shift) & 0x3F;
        u32 imms = 63 - shift;
        Emit32(0xD3400000 | (immr << 16) | (imms << 10) | (u32(src) << 5) | u32(dst));
    }

    // ── LSR Xd, Xn, #shift (logical shift right immediate) ──
    // Alias for UBFM Xd, Xn, #shift, #63.
    // Encoding (UBFM 64-bit): 0xD3400000 | (shift << 16) | (63 << 10) | (Xn << 5) | Xd
    void LsrImm(Arm64Reg dst, Arm64Reg src, u32 shift) {
        Emit32(0xD3400000 | ((shift & 0x3F) << 16) | (63 << 10) |
               (u32(src) << 5) | u32(dst));
    }

    // ── NEG Xd, Xn (negate) ──
    // Alias for SUB Xd, XZR, Xn.
    // Encoding: 0xCB000000 | (Xn << 16) | (ZR << 5) | Xd
    void Neg(Arm64Reg dst, Arm64Reg src) {
        Emit32(0xCB000000 | (u32(src) << 16) | (u32(ZR) << 5) | u32(dst));
    }

    // ── UBFX Xd, Xn, #lsb, #width (unsigned bitfield extract) ──
    // Extracts `width` bits from Xn starting at bit `lsb` and zero-extends
    // them into Xd. Alias for UBFM Xd, Xn, #lsb, #(lsb + width - 1).
    // Used to extract individual flag bits from NZCV during flag conversion.
    // Encoding (UBFM 64-bit): 0xD3400000 | (lsb << 16) | (imms << 10) | (Xn << 5) | Xd
    void Ubfx(Arm64Reg dst, Arm64Reg src, u32 lsb, u32 width) {
        u32 imms = lsb + width - 1;
        Emit32(0xD3400000 | ((lsb & 0x3F) << 16) | ((imms & 0x3F) << 10) |
               (u32(src) << 5) | u32(dst));
    }

    // ── LDR Xt, [Xn, Xm] (register offset, no shift) ──
    // 64-bit load: Xt = *(Xn + Xm). Used for memory operands with both base
    // and index registers.
    // Encoding: 0xF8406800 | (Xm << 16) | (Xn << 5) | Xt
    // (size=11, opc=01 for LDR, option=011 LSL, S=0, load=10)
    void Ldr64Reg(Arm64Reg dst, Arm64Reg base, Arm64Reg index) {
        Emit32(0xF8406800 | (u32(index) << 16) | (u32(base) << 5) | u32(dst));
    }

    // ── STR Xt, [Xn, Xm] (register offset, no shift) ──
    // 64-bit store: *(Xn + Xm) = Xt.
    // Encoding: 0xF8006000 | (Xm << 16) | (Xn << 5) | Xt
    // (size=11, opc=00 for STR, option=011 LSL, S=0, store=00)
    void Str64Reg(Arm64Reg src, Arm64Reg base, Arm64Reg index) {
        Emit32(0xF8006000 | (u32(index) << 16) | (u32(base) << 5) | u32(src));
    }

    // ── MADD Xd, Xn, Xm, Xa (multiply-add) ──
    // Computes Xd = Xa + Xn * Xm. Used for [base + index*scale] address
    // computation: MADD Xaddr, Xindex, Xscale, Xbase gives
    // Xaddr = Xbase + Xindex * Xscale.
    // Encoding: 0x9B000000 | (Xm << 16) | (Xa << 10) | (Xn << 5) | Xd
    void Madd(Arm64Reg dst, Arm64Reg src1, Arm64Reg src2, Arm64Reg src3) {
        Emit32(0x9B000000 | (u32(src2) << 16) | (u32(src3) << 10) |
               (u32(src1) << 5) | u32(dst));
    }

    // ── ADD Xd, Xn, Xm (register, explicit name) ──
    // Same as Add() but with a more explicit name for code that wants to make
    // the "register-register add" form obvious at the call site.
    // Encoding: 0x8B000000 | (Xm << 16) | (Xn << 5) | Xd
    void AddReg(Arm64Reg dst, Arm64Reg lhs, Arm64Reg rhs) {
        Emit32(0x8B000000 | (u32(rhs) << 16) | (u32(lhs) << 5) | u32(dst));
    }

    // ── ORR Xd, Xn, #imm12 (immediate, simplified) ──
    // The imm12 is placed in the imms field with N=0 and immr=0. The encoded
    // immediate value is determined by the N:immr:imms bitfield pattern, so
    // callers should pass values matching a real logical-immediate pattern
    // when correctness is required (the JIT uses register-form ORR for flag
    // conversion to avoid this restriction).
    // Encoding: 0xB2000000 | (imm12 << 10) | (Xn << 5) | Xd
    void OrrImm(Arm64Reg dst, Arm64Reg src, u32 imm12) {
        Emit32(0xB2000000 | ((imm12 & 0xFFF) << 10) | (u32(src) << 5) | u32(dst));
    }

    // Get current code offset (in bytes, from the start of the buffer).
    // Used for computing branch targets before they're emitted.
    s32 current_offset() const {
        return (s32)m_code.size();
    }

private:
    std::vector<u8> m_code;
};

} // namespace Core::Cpu::Jit
