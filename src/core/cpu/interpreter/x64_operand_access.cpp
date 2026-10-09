// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later
//
// Operand read/write helpers for the x86-64 interpreter (and eventually
// the JIT's slow path).  These translate a ZydisDecodedOperand into a
// concrete value by either reading a GPR/XMM or computing the memory
// address (segment base + base + index*scale + disp) and going through
// the guest memory manager.
//
// We expose these as a separate TU from the main interpreter loop so the
// JIT can call them from its own slow path without dragging the whole
// dispatch loop into the JIT translation unit.

#include "core/cpu/interpreter/x64_interpreter_backend.h"

#include <Zydis/Zydis.h>
#include <Zydis/Decoder.h>
#include <Zydis/DecoderTypes.h>

#include <cstring>

#include "common/assert.h"
#include "common/logging/log.h"

namespace Core::Cpu {

namespace {

// Map a ZydisRegister to an index into X64CpuState::gpr[]. Returns 0
// (RAX) for any non-GPR register code, which is intentional — callers
// must check the operand type before calling GetRegValue/SetRegValue.
u8 GprIndexFromZydis(ZydisRegister reg) {
    // ZydisRegisterRAX..ZydisRegisterR15 are contiguous in the
    // ZYDIS_REGISTER_RAX..ZYDIS_REGISTER_R15 enumeration; we rely on
    // that layout here. See Zydis/Zydis/Register.h for the table.
    if (reg >= ZYDIS_REGISTER_RAX && reg <= ZYDIS_REGISTER_R15) {
        return static_cast<u8>(reg - ZYDIS_REGISTER_RAX);
    }
    return 0;
}

bool IsGpr(ZydisRegister reg) {
    return reg >= ZYDIS_REGISTER_RAX && reg <= ZYDIS_REGISTER_R15;
}

bool IsXmm(ZydisRegister reg) {
    return (reg >= ZYDIS_REGISTER_XMM0 && reg <= ZYDIS_REGISTER_XMM15) ||
           (reg >= ZYDIS_REGISTER_YMM0 && reg <= ZYDIS_REGISTER_YMM15);
}

u8 XmmIndexFromZydis(ZydisRegister reg) {
    if (reg >= ZYDIS_REGISTER_XMM0 && reg <= ZYDIS_REGISTER_XMM15) {
        return static_cast<u8>(reg - ZYDIS_REGISTER_XMM0);
    }
    if (reg >= ZYDIS_REGISTER_YMM0 && reg <= ZYDIS_REGISTER_YMM15) {
        return static_cast<u8>(reg - ZYDIS_REGISTER_YMM0);
    }
    return 0;
}

// Mask a u64 to the given operand size in bits.
inline u64 MaskBySize(u64 v, u32 size_bits) {
    switch (size_bits) {
    case 8:  return v & 0xFFULL;
    case 16: return v & 0xFFFFULL;
    case 32: return v & 0xFFFF'FFFFULL;
    case 64: return v;
    default: return v;
    }
}

// Sign-extend a u64 from `size_bits` to 64 bits.
inline u64 SignExtend(u64 v, u32 size_bits) {
    switch (size_bits) {
    case 8:  return static_cast<u64>(static_cast<s8>(v & 0xFF));
    case 16: return static_cast<u64>(static_cast<s16>(v & 0xFFFF));
    case 32: return static_cast<u64>(static_cast<s32>(v & 0xFFFF'FFFF));
    case 64: return v;
    default: return v;
    }
}

// Compute the segment-base override for memory operands. In long mode
// only FS/GS have a non-zero base; the others are treated as zero.
// Zydis already encodes the segment register in op->mem.segment.
u64 SegmentBase(const X64CpuState& state, ZydisRegister seg) {
    switch (seg) {
    case ZYDIS_REGISTER_FS: return state.fs_base;
    case ZYDIS_REGISTER_GS: return state.gs_base;
    case ZYDIS_REGISTER_CS: return state.cs_base;
    case ZYDIS_REGISTER_DS: return state.ds_base;
    case ZYDIS_REGISTER_ES: return state.es_base;
    case ZYDIS_REGISTER_SS: return state.ss_base;
    default: return 0;
    }
}

} // namespace

u64 GetRegValue(const X64CpuState& state, ZydisRegister reg, u32 operand_size) {
    if (reg == ZYDIS_REGISTER_RIP || reg == ZYDIS_REGISTER_EIP) {
        return state.rip;
    }
    if (IsGpr(reg)) {
        const u8 idx = GprIndexFromZydis(reg);
        const u64 v = state.gpr[idx];
        // The operand size determines whether the caller wants the
        // low byte, low word, low dword, or the full 64 bits.  We always
        // return the zero-extended slice.
        return MaskBySize(v, operand_size * 8);
    }
    if (IsXmm(reg)) {
        // GetRegValue is for integer operands; XMM uses a separate path.
        return 0;
    }
    switch (reg) {
    case ZYDIS_REGISTER_RFLAGS: return state.rflags;
    case ZYDIS_REGISTER_EFLAGS: return state.rflags & 0xFFFF'FFFF;
    case ZYDIS_REGISTER_FLAGS:  return state.rflags & 0xFFFF;
    }
    return 0;
}

void SetRegValue(X64CpuState& state, ZydisRegister reg, u64 value, u32 operand_size) {
    if (reg == ZYDIS_REGISTER_RIP || reg == ZYDIS_REGISTER_EIP) {
        // Caller wants to update RIP — set the bit width appropriately.
        if (operand_size == 4) {
            state.rip = value & 0xFFFF'FFFFULL;
        } else {
            state.rip = value;
        }
        return;
    }
    if (IsGpr(reg)) {
        const u8 idx = GprIndexFromZydis(reg);
        // For 8-bit GPR writes we have to deal with the high-byte aliasing
        // (AH/CH/DH/BH = the high byte of AX/CX/DX/BX when REX is absent).
        // Zydis handles this by giving us a different register code — we
        // just need to mask/merge the correct byte.
        if (operand_size == 1) {
            // The 8-bit sub-register codes (AL/CL/DL/BL/AH/CH/DH/BH and
            // their R8-R15 spl/r8b-r15b variants) cover the range
            // ZYDIS_REGISTER_AL..ZYDIS_REGISTER_R15B. The mapping isn't
            // a simple subtract because of the high-byte aliases; use a
            // small table.
            static const struct {
                ZydisRegister code;
                u8 gpr_idx;
                u8 shift; // 0 for low byte, 8 for high byte (AH/CH/DH/BH)
            } kByteRegs[] = {
                {ZYDIS_REGISTER_AL,  GPR_RAX, 0},
                {ZYDIS_REGISTER_CL,  GPR_RCX, 0},
                {ZYDIS_REGISTER_DL,  GPR_RDX, 0},
                {ZYDIS_REGISTER_BL,  GPR_RBX, 0},
                {ZYDIS_REGISTER_AH,  GPR_RAX, 8},
                {ZYDIS_REGISTER_CH,  GPR_RCX, 8},
                {ZYDIS_REGISTER_DH,  GPR_RDX, 8},
                {ZYDIS_REGISTER_BH,  GPR_RBX, 8},
                {ZYDIS_REGISTER_SPL, GPR_RSP, 0},
                {ZYDIS_REGISTER_BPL, GPR_RBP, 0},
                {ZYDIS_REGISTER_SIL, GPR_RSI, 0},
                {ZYDIS_REGISTER_DIL, GPR_RDI, 0},
                {ZYDIS_REGISTER_R8B,  GPR_R8,  0},
                {ZYDIS_REGISTER_R9B,  GPR_R9,  0},
                {ZYDIS_REGISTER_R10B, GPR_R10, 0},
                {ZYDIS_REGISTER_R11B, GPR_R11, 0},
                {ZYDIS_REGISTER_R12B, GPR_R12, 0},
                {ZYDIS_REGISTER_R13B, GPR_R13, 0},
                {ZYDIS_REGISTER_R14B, GPR_R14, 0},
                {ZYDIS_REGISTER_R15B, GPR_R15, 0},
            };
            for (const auto& entry : kByteRegs) {
                if (entry.code == reg) {
                    const u64 mask = 0xFFULL << entry.shift;
                    state.gpr[entry.gpr_idx] =
                        (state.gpr[entry.gpr_idx] & ~mask) |
                        ((value << entry.shift) & mask);
                    return;
                }
            }
            // Unknown 8-bit register — shouldn't happen; log and bail.
            return;
        }
        // 16/32/64-bit: zero-extend the value into the GPR (writing EAX
        // zeros the high 32 bits of RAX in long mode; writing AX leaves
        // the high 48 bits untouched).
        if (operand_size == 8) {
            state.gpr[idx] = value;
        } else if (operand_size == 4) {
            state.gpr[idx] = value & 0xFFFF'FFFFULL;
        } else { // 2 bytes
            state.gpr[idx] = (state.gpr[idx] & ~0xFFFFULL) | (value & 0xFFFFULL);
        }
        return;
    }
    // RFLAGS writes — keep the low 32 bits only.
    if (reg == ZYDIS_REGISTER_RFLAGS) {
        state.rflags = value;
        return;
    }
}

// Read `size` bytes from guest memory at `guest_addr`. Sign-extends if
// `is_signed` is true.
//
// In shadPS4's memory model, guest virtual addresses are mapped 1:1
// into the host process's address space (the loader maps the ELF
// segments with mmap at fixed guest VAddrs), so we can just
// reinterpret_cast the guest address to a host pointer. This matches
// what the rest of the emulator does — see e.g. src/core/cpu_patches.cpp
// for the same pattern.
u64 ReadMemory(u64 guest_addr, u32 size, bool is_signed) {
    // Guard against null/invalid addresses that would cause SIGSEGV.
    // Guest memory is at 0x200000000-0x900000000. Addresses outside
    // this range are invalid and should return 0 instead of crashing.
    if (guest_addr < 0x100000000ULL || guest_addr >= 0x900000000ULL) {
        LOG_WARNING(Core_Cpu, "ReadMemory: invalid guest_addr=0x{:x} (size={})", guest_addr, size);
        return 0;
    }
    const u8* p = reinterpret_cast<const u8*>(guest_addr);
    u64 value = 0;
    switch (size) {
    case 1: std::memcpy(&value, p, 1); value &= 0xFFULL;             break;
    case 2: std::memcpy(&value, p, 2); value &= 0xFFFFULL;           break;
    case 4: std::memcpy(&value, p, 4); value &= 0xFFFF'FFFFULL;       break;
    case 8: std::memcpy(&value, p, 8);                                break;
    default:
        LOG_ERROR(Core_Cpu, "ReadMemory: unsupported size {}", size);
        return 0;
    }
    if (is_signed) {
        switch (size) {
        case 1: value = static_cast<u64>(static_cast<s8>(value));   break;
        case 2: value = static_cast<u64>(static_cast<s16>(value)); break;
        case 4: value = static_cast<u64>(static_cast<s32>(value)); break;
        }
    }
    return value;
}

void WriteMemory(u64 guest_addr, u64 value, u32 size, bool /*is_signed*/) {
    // Guard against null/invalid addresses.
    if (guest_addr < 0x100000000ULL || guest_addr >= 0x900000000ULL) {
        LOG_WARNING(Core_Cpu, "WriteMemory: invalid guest_addr=0x{:x} (size={})", guest_addr, size);
        return;
    }
    u8* p = reinterpret_cast<u8*>(guest_addr);
    switch (size) {
    case 1: { const u8 v  = static_cast<u8>(value);  std::memcpy(p, &v, 1); break; }
    case 2: { const u16 v = static_cast<u16>(value); std::memcpy(p, &v, 2); break; }
    case 4: { const u32 v = static_cast<u32>(value); std::memcpy(p, &v, 4); break; }
    case 8: { std::memcpy(p, &value, 8);                                break; }
    default:
        LOG_ERROR(Core_Cpu, "WriteMemory: unsupported size {}", size);
        break;
    }
}

// Compute the linear guest virtual address of a memory operand.
static u64 ComputeMemAddr(const X64CpuState& state,
                          const ZydisDecodedInstruction& inst,
                          const ZydisDecodedOperand& op) {
    const auto& mem = op.mem;
    u64 base = 0;
    if (mem.base != ZYDIS_REGISTER_NONE) {
        if (mem.base == ZYDIS_REGISTER_RIP || mem.base == ZYDIS_REGISTER_EIP) {
            // RIP-relative addressing: Zydis encodes the displacement
            // relative to the address of the NEXT instruction
            // (RIP + instruction.length), not the current RIP.
            base = state.rip + inst.length;
        } else {
            base = GetRegValue(state, mem.base, 8);
        }
    }
    u64 index = 0;
    if (mem.index != ZYDIS_REGISTER_NONE) {
        index = GetRegValue(state, mem.index, 8);
        // The scale is 1/2/4/8 (encoded as 0/1/2/3 in Zydis).
        index *= (1ULL << mem.scale);
    }
    u64 disp = static_cast<u64>(static_cast<s64>(mem.disp.value));
    u64 seg_base = SegmentBase(state, mem.segment);
    return seg_base + base + index + disp;
}

u64 ReadOperand(const X64CpuState& state,
                const ZydisDecodedInstruction& inst,
                const ZydisDecodedOperand& op,
                bool is_signed) {
    const auto& o = static_cast<const ZydisDecodedOperand&>(op);
    switch (o.type) {
    case ZYDIS_OPERAND_TYPE_REGISTER:
        return GetRegValue(state, o.reg.value, o.size / 8);
    case ZYDIS_OPERAND_TYPE_IMMEDIATE:
        if (o.imm.is_signed) {
            return static_cast<u64>(static_cast<s64>(o.imm.value.s));
        }
        return static_cast<u64>(o.imm.value.u);
    case ZYDIS_OPERAND_TYPE_MEMORY: {
        const u64 addr = ComputeMemAddr(state, inst, op);
        return ReadMemory(addr, o.size / 8, is_signed);
    }
    case ZYDIS_OPERAND_TYPE_POINTER:
        // Far pointer — never used in long-mode user space. Return the
        // offset field (which is what callers usually want).
        return static_cast<u64>(o.ptr.offset);
    default:
        return 0;
    }
}

void WriteOperand(X64CpuState& state,
                  const ZydisDecodedInstruction& inst,
                  const ZydisDecodedOperand& op,
                  u64 value,
                  bool /*is_signed*/) {
    const auto& o = static_cast<const ZydisDecodedOperand&>(op);
    switch (o.type) {
    case ZYDIS_OPERAND_TYPE_REGISTER:
        SetRegValue(state, o.reg.value, value, o.size / 8);
        break;
    case ZYDIS_OPERAND_TYPE_MEMORY: {
        const u64 addr = ComputeMemAddr(state, inst, op);
        WriteMemory(addr, value, o.size / 8, false);
        break;
    }
    default:
        // IMMEDIATE / POINTER operands can't be written to; ignore.
        break;
    }
}

} // namespace Core::Cpu
