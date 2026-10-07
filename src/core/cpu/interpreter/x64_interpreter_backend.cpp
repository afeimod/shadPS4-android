// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later
//
// X64InterpreterBackend — the main instruction-by-instruction x86-64
// interpreter.
//
// This file implements:
//   1. The Execute() entry point: sets up an X64CpuState, jumps into the
//      main loop, decodes one instruction at a time, dispatches to a
//      per-mnemonic handler, advances RIP, repeats.
//   2. The dispatch table for the first wave of ~30 mnemonics. Many more
//      will be added as we hit them in real PS4 homebrew / games.
//   3. The flags-update helpers and condition-code evaluator.

#include "core/cpu/interpreter/x64_interpreter_backend.h"

#include <Zydis/Zydis.h>
#include <Zydis/Decoder.h>
#include <Zydis/DecoderTypes.h>
#include <Zydis/Mnemonic.h>
#include <Zydis/Register.h>

#include <array>
#include <atomic>
#include <cstring>
#include <functional>

#if defined(__ANDROID__)
#include <sys/mman.h>
#include <cerrno>
#endif

#include "common/assert.h"
#include "common/logging/log.h"

namespace Core::Cpu {

// ─────────────────────────────────────────────────────────────────────────
//  Flags helpers (defined here, not in x64_operand_access.cpp, so they
//  live next to the dispatcher that uses them).
// ─────────────────────────────────────────────────────────────────────────

namespace {

inline void SetFlag(X64CpuState& s, u64 mask, bool on) {
    if (on) {
        s.rflags |= mask;
    } else {
        s.rflags &= ~mask;
    }
}

inline bool GetFlag(const X64CpuState& s, u64 mask) {
    return (s.rflags & mask) != 0;
}

// Parity lookup — for the PF flag, Intel checks the low 8 bits of the
// result and sets PF=1 if the number of set bits is *even*.
bool ParityEven(u8 v) {
    static const bool kTable[256] = {
        true,  false, false, true,  false, true,  true,  false,
        false, true,  true,  false, true,  false, false, true,
        false, true,  true,  false, true,  false, false, true,
        true,  false, false, true,  false, true,  true,  false,
        false, true,  true,  false, true,  false, false, true,
        true,  false, false, true,  false, true,  true,  false,
        true,  false, false, true,  false, true,  true,  false,
        false, true,  true,  false, true,  false, false, true,
        false, true,  true,  false, true,  false, false, true,
        true,  false, false, true,  false, true,  true,  false,
        true,  false, false, true,  false, true,  true,  false,
        false, true,  true,  false, true,  false, false, true,
        true,  false, false, true,  false, true,  true,  false,
        false, true,  true,  false, true,  false, false, true,
        false, true,  true,  false, true,  false, false, true,
        true,  false, false, true,  false, true,  true,  false,
        true,  false, false, true,  false, true,  true,  false,
        false, true,  true,  false, true,  false, false, true,
        false, true,  true,  false, true,  false, false, true,
        true,  false, false, true,  false, true,  true,  false,
        true,  false, false, true,  false, true,  true,  false,
        false, true,  true,  false, true,  false, false, true,
        false, true,  true,  false, true,  false, false, true,
        true,  false, false, true,  false, true,  true,  false,
        true,  false, false, true,  false, true,  true,  false,
        false, true,  true,  false, true,  false, false, true,
        false, true,  true,  false, true,  false, false, true,
        true,  false, false, true,  false, true,  true,  false,
        false, true,  true,  false, true,  false, false, true,
        true,  false, false, true,  false, true,  true,  false,
        true,  false, false, true,  false, true,  true,  false,
        false, true,  true,  false, true,  false, false, true,
    };
    return kTable[v];
}

inline u64 SignBit(u32 size_bits) {
    if (size_bits == 64) return 1ULL << 63;
    return 1ULL << (size_bits - 1);
}

inline u64 OperandMask(u32 size_bits) {
    if (size_bits == 64) return ~0ULL;
    return (1ULL << size_bits) - 1;
}

} // namespace

void UpdateFlagsAdd(X64CpuState& s, u64 lhs, u64 rhs, u64 result, u32 size_bits) {
    const u64 mask = OperandMask(size_bits);
    const u64 sign = SignBit(size_bits);
    const u64 r = result & mask;
    SetFlag(s, RflagsBits::CF, result > mask);
    SetFlag(s, RflagsBits::PF, ParityEven(static_cast<u8>(r)));
    SetFlag(s, RflagsBits::AF, ((lhs ^ rhs ^ r) & 0x10) != 0);
    SetFlag(s, RflagsBits::ZF, r == 0);
    SetFlag(s, RflagsBits::SF, (r & sign) != 0);
    SetFlag(s, RflagsBits::OF, ((~(lhs ^ rhs) & (lhs ^ r)) & sign) != 0);
}

void UpdateFlagsSub(X64CpuState& s, u64 lhs, u64 rhs, u64 result, u32 size_bits) {
    const u64 mask = OperandMask(size_bits);
    const u64 sign = SignBit(size_bits);
    const u64 r = result & mask;
    SetFlag(s, RflagsBits::CF, lhs < rhs);
    SetFlag(s, RflagsBits::PF, ParityEven(static_cast<u8>(r)));
    SetFlag(s, RflagsBits::AF, ((lhs ^ rhs ^ r) & 0x10) != 0);
    SetFlag(s, RflagsBits::ZF, r == 0);
    SetFlag(s, RflagsBits::SF, (r & sign) != 0);
    SetFlag(s, RflagsBits::OF, (((lhs ^ rhs) & (lhs ^ r)) & sign) != 0);
}

void UpdateFlagsAdc(X64CpuState& s, u64 lhs, u64 rhs, bool carry_in, u64 result, u32 size_bits) {
    const u64 mask = OperandMask(size_bits);
    const u64 sign = SignBit(size_bits);
    const u64 r = result & mask;
    const u64 carry_in_u = carry_in ? 1 : 0;
    SetFlag(s, RflagsBits::CF, (result > mask) || (carry_in_u && r == (lhs + rhs) & mask));
    SetFlag(s, RflagsBits::PF, ParityEven(static_cast<u8>(r)));
    SetFlag(s, RflagsBits::AF, ((lhs ^ rhs ^ r) & 0x10) != 0);
    SetFlag(s, RflagsBits::ZF, r == 0);
    SetFlag(s, RflagsBits::SF, (r & sign) != 0);
    SetFlag(s, RflagsBits::OF, ((~(lhs ^ rhs) & (lhs ^ r)) & sign) != 0);
}

void UpdateFlagsSbb(X64CpuState& s, u64 lhs, u64 rhs, bool carry_in, u64 result, u32 size_bits) {
    const u64 mask = OperandMask(size_bits);
    const u64 sign = SignBit(size_bits);
    const u64 r = result & mask;
    const u64 borrow_in = carry_in ? 1 : 0;
    SetFlag(s, RflagsBits::CF, lhs < (rhs + borrow_in));
    SetFlag(s, RflagsBits::PF, ParityEven(static_cast<u8>(r)));
    SetFlag(s, RflagsBits::AF, ((lhs ^ rhs ^ r) & 0x10) != 0);
    SetFlag(s, RflagsBits::ZF, r == 0);
    SetFlag(s, RflagsBits::SF, (r & sign) != 0);
    SetFlag(s, RflagsBits::OF, (((lhs ^ rhs) & (lhs ^ r)) & sign) != 0);
}

void UpdateFlagsLogic(X64CpuState& s, u64 result, u32 size_bits) {
    const u64 mask = OperandMask(size_bits);
    const u64 sign = SignBit(size_bits);
    const u64 r = result & mask;
    SetFlag(s, RflagsBits::CF, false);
    SetFlag(s, RflagsBits::OF, false);
    SetFlag(s, RflagsBits::PF, ParityEven(static_cast<u8>(r)));
    SetFlag(s, RflagsBits::AF, false); // undefined, set to 0
    SetFlag(s, RflagsBits::ZF, r == 0);
    SetFlag(s, RflagsBits::SF, (r & sign) != 0);
}

void UpdateFlagsInc(X64CpuState& s, u64 lhs, u64 result, u32 size_bits) {
    // INC preserves CF.
    const u64 mask = OperandMask(size_bits);
    const u64 sign = SignBit(size_bits);
    const u64 r = result & mask;
    SetFlag(s, RflagsBits::PF, ParityEven(static_cast<u8>(r)));
    SetFlag(s, RflagsBits::AF, ((lhs ^ 1 ^ r) & 0x10) != 0);
    SetFlag(s, RflagsBits::ZF, r == 0);
    SetFlag(s, RflagsBits::SF, (r & sign) != 0);
    SetFlag(s, RflagsBits::OF, ((~(lhs ^ 1) & (lhs ^ r)) & sign) != 0);
}

void UpdateFlagsDec(X64CpuState& s, u64 lhs, u64 result, u32 size_bits) {
    // DEC preserves CF.
    const u64 mask = OperandMask(size_bits);
    const u64 sign = SignBit(size_bits);
    const u64 r = result & mask;
    SetFlag(s, RflagsBits::PF, ParityEven(static_cast<u8>(r)));
    SetFlag(s, RflagsBits::AF, ((lhs ^ 1 ^ r) & 0x10) != 0);
    SetFlag(s, RflagsBits::ZF, r == 0);
    SetFlag(s, RflagsBits::SF, (r & sign) != 0);
    SetFlag(s, RflagsBits::OF, (((lhs ^ 1) & (lhs ^ r)) & sign) != 0);
}

bool EvaluateConditionCode(const X64CpuState& s, ZydisMnemonic_ mnemonic) {
    const bool cf = GetFlag(s, RflagsBits::CF);
    const bool pf = GetFlag(s, RflagsBits::PF);
    const bool zf = GetFlag(s, RflagsBits::ZF);
    const bool sf = GetFlag(s, RflagsBits::SF);
    const bool of = GetFlag(s, RflagsBits::OF);
    switch (mnemonic) {
    case ZYDIS_MNEMONIC_JO:  return of;
    case ZYDIS_MNEMONIC_JNO: return !of;
    case ZYDIS_MNEMONIC_JB:  return cf;
    case ZYDIS_MNEMONIC_JNB: return !cf;
    case ZYDIS_MNEMONIC_JZ:  return zf;
    case ZYDIS_MNEMONIC_JNZ: return !zf;
    case ZYDIS_MNEMONIC_JBE: return cf || zf;
    case ZYDIS_MNEMONIC_JNBE:return !cf && !zf;
    case ZYDIS_MNEMONIC_JS:  return sf;
    case ZYDIS_MNEMONIC_JNS: return !sf;
    case ZYDIS_MNEMONIC_JP:  return pf;
    case ZYDIS_MNEMONIC_JNP: return !pf;
    case ZYDIS_MNEMONIC_JL:  return sf != of;
    case ZYDIS_MNEMONIC_JNL: return sf == of;
    case ZYDIS_MNEMONIC_JLE: return zf || (sf != of);
    case ZYDIS_MNEMONIC_JNLE:return !zf && (sf == of);
    case ZYDIS_MNEMONIC_JCXZ:return (s.gpr[GPR_RCX] & 0xFFFF) == 0;
    case ZYDIS_MNEMONIC_JECXZ:return (s.gpr[GPR_RCX] & 0xFFFF'FFFFULL) == 0;
    case ZYDIS_MNEMONIC_JRCXZ:return s.gpr[GPR_RCX] == 0;
    default:
        LOG_ERROR(Core_Cpu, "EvaluateConditionCode: mnemonic {} not a Jcc",
                  static_cast<unsigned>(mnemonic));
        return false;
    }
}

// ─────────────────────────────────────────────────────────────────────────
//  Diagnostics
// ─────────────────────────────────────────────────────────────────────────

namespace {
struct TraceEntry {
    u64 rip;
    ZydisMnemonic_ mnemonic;
};
constexpr unsigned int kTraceRingSize = 64;
TraceEntry g_trace_ring[kTraceRingSize];
std::atomic<unsigned int> g_trace_ring_pos{0};
std::atomic<unsigned int> g_trace_ring_count{0};

void RecordTrace(u64 rip, ZydisMnemonic_ mn) {
    const unsigned int idx = g_trace_ring_pos.fetch_add(1) % kTraceRingSize;
    g_trace_ring[idx] = {rip, mn};
    unsigned int count = g_trace_ring_count.load();
    while (count < kTraceRingSize) {
        if (g_trace_ring_count.compare_exchange_weak(count, count + 1)) break;
    }
}
} // namespace

void DumpRecentInterpreterTrace(const char* tag, unsigned int count) {
    const unsigned int have = g_trace_ring_count.load();
    const unsigned int dump = std::min(have, count);
    LOG_ERROR(Core_Cpu, "Interpreter trace dump [{}]: {} entries", tag ? tag : "", dump);
    const unsigned int start_pos = g_trace_ring_pos.load();
    for (unsigned int i = 0; i < dump; ++i) {
        const unsigned int idx = (start_pos + kTraceRingSize - dump + i) % kTraceRingSize;
        const TraceEntry& e = g_trace_ring[idx];
        LOG_ERROR(Core_Cpu, "  [{:3}] rip=0x{:016x} mnem={}", i, e.rip, static_cast<unsigned>(e.mnemonic));
    }
}

// ─────────────────────────────────────────────────────────────────────────
//  Per-thread guest-state hint (used by signal handler to dump trace
//  on a segfault)
// ─────────────────────────────────────────────────────────────────────────

namespace {
struct ThreadHint {
    u64 rip;
    u64 stack_top;
};
thread_local ThreadHint tls_thread_hint{0, 0};
} // namespace

void X64InterpreterBackend::SetInterpreterThreadStackHint(u64 guest_rip, u64 stack_top) {
    tls_thread_hint = {guest_rip, stack_top};
}

// ─────────────────────────────────────────────────────────────────────────
//  Backend Impl
// ─────────────────────────────────────────────────────────────────────────

struct X64InterpreterBackend::Impl {
    ZydisDecoder decoder{};
    X64InterpreterRuntimeConfig config{};
    std::function<NativeBlockLookup(u64)> native_block_provider;

    Impl() {
        // Initialize the Zydis decoder for 64-bit mode (long mode).
        ZydisDecoderInit(&decoder, ZYDIS_MACHINE_MODE_LONG_64, ZYDIS_STACK_WIDTH_64);
    }
};

X64InterpreterBackend::X64InterpreterBackend() : m_impl(std::make_unique<Impl>()) {}
X64InterpreterBackend::~X64InterpreterBackend() = default;

void X64InterpreterBackend::SetRuntimeConfig(const X64InterpreterRuntimeConfig& cfg) {
    m_impl->config = cfg;
}

void X64InterpreterBackend::SetNativeBlockProvider(
    std::function<NativeBlockLookup(u64)> provider) {
    m_impl->native_block_provider = std::move(provider);
}

// Forward decls of the handler functions. Each handler takes the
// decoded instruction and the current CPU state, returns the new RIP
// (in case of a taken branch) or `state.rip + inst.length` (default
// fall-through). When a handler wants to exit the loop entirely (e.g.
// HLE call), it sets `state.in_guest_code = false`.
using Handler = u64 (*)(X64CpuState& s, const ZydisDecodedInstruction& inst, const ZydisDecodedOperand* operands);

// Handler table — indexed by ZydisMnemonic. Big switch is in the .cpp;
// we declare each handler separately below.
#define HANDLER(name)                                                          \
    static u64 Handle_##name(X64CpuState& s, const ZydisDecodedInstruction& inst, \
                              const ZydisDecodedOperand* operands)

HANDLER(Mov);
HANDLER(Push);
HANDLER(Pop);
HANDLER(Lea);
HANDLER(Add);
HANDLER(Sub);
HANDLER(And);
HANDLER(Or);
HANDLER(Xor);
HANDLER(Test);
HANDLER(Cmp);
HANDLER(Inc);
HANDLER(Dec);
HANDLER(Not);
HANDLER(Neg);
HANDLER(Jmp);
HANDLER(Jcc);
HANDLER(Call);
HANDLER(Ret);
HANDLER(Nop);
HANDLER(Xchg);
HANDLER(Movsx);
HANDLER(Movzx);
HANDLER(Cmov);
HANDLER(Setcc);
HANDLER(Shl);
HANDLER(Shr);
HANDLER(Sar);
HANDLER(Mul);
HANDLER(Imul);
HANDLER(Div);
HANDLER(Idiv);
HANDLER(Xadd);
HANDLER(CmpXchg);

// Dispatcher: looks up the mnemonic in the handler table. Returns
// nullptr for unimplemented mnemonics — the caller (the main loop) then
// logs and skips (or aborts if config.abort_on_unknown is set).
static Handler LookupHandler(ZydisMnemonic_ mn) {
    switch (mn) {
    case ZYDIS_MNEMONIC_MOV:    return Handle_Mov;
    case ZYDIS_MNEMONIC_PUSH:   return Handle_Push;
    case ZYDIS_MNEMONIC_POP:    return Handle_Pop;
    case ZYDIS_MNEMONIC_LEA:    return Handle_Lea;
    case ZYDIS_MNEMONIC_ADD:    return Handle_Add;
    case ZYDIS_MNEMONIC_SUB:    return Handle_Sub;
    case ZYDIS_MNEMONIC_AND:    return Handle_And;
    case ZYDIS_MNEMONIC_OR:     return Handle_Or;
    case ZYDIS_MNEMONIC_XOR:    return Handle_Xor;
    case ZYDIS_MNEMONIC_TEST:   return Handle_Test;
    case ZYDIS_MNEMONIC_CMP:    return Handle_Cmp;
    case ZYDIS_MNEMONIC_INC:    return Handle_Inc;
    case ZYDIS_MNEMONIC_DEC:    return Handle_Dec;
    case ZYDIS_MNEMONIC_NOT:    return Handle_Not;
    case ZYDIS_MNEMONIC_NEG:    return Handle_Neg;
    case ZYDIS_MNEMONIC_JMP:    return Handle_Jmp;
    // Jcc — handled by Handle_Jcc which evaluates the condition itself.
    case ZYDIS_MNEMONIC_JO: case ZYDIS_MNEMONIC_JNO:
    case ZYDIS_MNEMONIC_JB:  case ZYDIS_MNEMONIC_JNB:
    case ZYDIS_MNEMONIC_JZ:  case ZYDIS_MNEMONIC_JNZ:
    case ZYDIS_MNEMONIC_JBE: case ZYDIS_MNEMONIC_JNBE:
    case ZYDIS_MNEMONIC_JS:  case ZYDIS_MNEMONIC_JNS:
    case ZYDIS_MNEMONIC_JP:  case ZYDIS_MNEMONIC_JNP:
    case ZYDIS_MNEMONIC_JL:  case ZYDIS_MNEMONIC_JNL:
    case ZYDIS_MNEMONIC_JLE: case ZYDIS_MNEMONIC_JNLE:
    case ZYDIS_MNEMONIC_JCXZ:case ZYDIS_MNEMONIC_JECXZ:case ZYDIS_MNEMONIC_JRCXZ:
        return Handle_Jcc;
    case ZYDIS_MNEMONIC_CALL:  return Handle_Call;
    case ZYDIS_MNEMONIC_RET:   return Handle_Ret;
    case ZYDIS_MNEMONIC_NOP:   return Handle_Nop;
    case ZYDIS_MNEMONIC_XCHG:  return Handle_Xchg;
    case ZYDIS_MNEMONIC_MOVSX: return Handle_Movsx;
    case ZYDIS_MNEMONIC_MOVZX: return Handle_Movzx;
    case ZYDIS_MNEMONIC_CMOVO: case ZYDIS_MNEMONIC_CMOVNO:
    case ZYDIS_MNEMONIC_CMOVB: case ZYDIS_MNEMONIC_CMOVNB:
    case ZYDIS_MNEMONIC_CMOVZ: case ZYDIS_MNEMONIC_CMOVNZ:
    case ZYDIS_MNEMONIC_CMOVBE:case ZYDIS_MNEMONIC_CMOVNBE:
    case ZYDIS_MNEMONIC_CMOVS: case ZYDIS_MNEMONIC_CMOVNS:
    case ZYDIS_MNEMONIC_CMOVP: case ZYDIS_MNEMONIC_CMOVNP:
    case ZYDIS_MNEMONIC_CMOVL: case ZYDIS_MNEMONIC_CMOVNL:
    case ZYDIS_MNEMONIC_CMOVLE:case ZYDIS_MNEMONIC_CMOVNLE:
        return Handle_Cmov;
    case ZYDIS_MNEMONIC_SETO: case ZYDIS_MNEMONIC_SETNO:
    case ZYDIS_MNEMONIC_SETB: case ZYDIS_MNEMONIC_SETNB:
    case ZYDIS_MNEMONIC_SETZ: case ZYDIS_MNEMONIC_SETNZ:
    case ZYDIS_MNEMONIC_SETBE:case ZYDIS_MNEMONIC_SETNBE:
    case ZYDIS_MNEMONIC_SETS: case ZYDIS_MNEMONIC_SETNS:
    case ZYDIS_MNEMONIC_SETP: case ZYDIS_MNEMONIC_SETNP:
    case ZYDIS_MNEMONIC_SETL: case ZYDIS_MNEMONIC_SETNL:
    case ZYDIS_MNEMONIC_SETLE:case ZYDIS_MNEMONIC_SETNLE:
        return Handle_Setcc;
    case ZYDIS_MNEMONIC_SHL: return Handle_Shl;
    case ZYDIS_MNEMONIC_SHR:                         return Handle_Shr;
    case ZYDIS_MNEMONIC_SAR:                         return Handle_Sar;
    case ZYDIS_MNEMONIC_MUL:  return Handle_Mul;
    case ZYDIS_MNEMONIC_IMUL: return Handle_Imul;
    case ZYDIS_MNEMONIC_DIV:  return Handle_Div;
    case ZYDIS_MNEMONIC_IDIV: return Handle_Idiv;
    case ZYDIS_MNEMONIC_XADD: return Handle_Xadd;
    case ZYDIS_MNEMONIC_CMPXCHG: return Handle_CmpXchg;
    default:                  return nullptr;
    }
}

// ─────────────────────────────────────────────────────────────────────────
//  Handler implementations
// ─────────────────────────────────────────────────────────────────────────

HANDLER(Mov) {
    const u64 src = ReadOperand(s, inst, operands[1], /*is_signed=*/false);
    WriteOperand(s, inst, operands[0], src, /*is_signed=*/false);
    return s.rip + inst.length;
}

HANDLER(Push) {
    const u64 v = ReadOperand(s, inst, operands[0], /*is_signed=*/false);
    const u32 sz = operands[0].size / 8;
    s.gpr[GPR_RSP] -= sz;
    WriteMemory(s.gpr[GPR_RSP], v, sz, false);
    return s.rip + inst.length;
}

HANDLER(Pop) {
    const u32 sz = operands[0].size / 8;
    const u64 v = ReadMemory(s.gpr[GPR_RSP], sz, false);
    WriteOperand(s, inst, operands[0], v, false);
    s.gpr[GPR_RSP] += sz;
    return s.rip + inst.length;
}

HANDLER(Lea) {
    const auto& op = operands[1];
    if (op.type != ZYDIS_OPERAND_TYPE_MEMORY) {
        LOG_ERROR(Core_Cpu, "LEA: src not memory");
        return s.rip + inst.length;
    }
    const u64 addr = op.mem.base != ZYDIS_REGISTER_NONE
                         ? GetRegValue(s, op.mem.base, 8)
                         : 0;
    const u64 index = op.mem.index != ZYDIS_REGISTER_NONE
                          ? GetRegValue(s, op.mem.index, 8) * (1ULL << op.mem.scale)
                          : 0;
    const u64 disp = static_cast<u64>(static_cast<s64>(op.mem.disp.value));
    // LEA ignores segment overrides — it just gives you the linear
    // address arithmetic.
    const u64 effective = addr + index + disp;
    WriteOperand(s, inst, operands[0], effective, false);
    return s.rip + inst.length;
}

HANDLER(Add) {
    const u64 lhs = ReadOperand(s, inst, operands[0], false);
    const u64 rhs = ReadOperand(s, inst, operands[1], false);
    const u32 sz = operands[0].size / 8;
    const u64 result = (lhs + rhs) & OperandMask(sz * 8);
    WriteOperand(s, inst, operands[0], result, false);
    UpdateFlagsAdd(s, lhs, rhs, lhs + rhs, sz * 8);
    return s.rip + inst.length;
}

HANDLER(Sub) {
    const u64 lhs = ReadOperand(s, inst, operands[0], false);
    const u64 rhs = ReadOperand(s, inst, operands[1], false);
    const u32 sz = operands[0].size / 8;
    const u64 result = (lhs - rhs) & OperandMask(sz * 8);
    WriteOperand(s, inst, operands[0], result, false);
    UpdateFlagsSub(s, lhs, rhs, lhs - rhs, sz * 8);
    return s.rip + inst.length;
}

HANDLER(And) {
    const u64 lhs = ReadOperand(s, inst, operands[0], false);
    const u64 rhs = ReadOperand(s, inst, operands[1], false);
    const u32 sz = operands[0].size / 8;
    const u64 r = lhs & rhs;
    WriteOperand(s, inst, operands[0], r & OperandMask(sz * 8), false);
    UpdateFlagsLogic(s, r, sz * 8);
    return s.rip + inst.length;
}

HANDLER(Or) {
    const u64 lhs = ReadOperand(s, inst, operands[0], false);
    const u64 rhs = ReadOperand(s, inst, operands[1], false);
    const u32 sz = operands[0].size / 8;
    const u64 r = lhs | rhs;
    WriteOperand(s, inst, operands[0], r & OperandMask(sz * 8), false);
    UpdateFlagsLogic(s, r, sz * 8);
    return s.rip + inst.length;
}

HANDLER(Xor) {
    const u64 lhs = ReadOperand(s, inst, operands[0], false);
    const u64 rhs = ReadOperand(s, inst, operands[1], false);
    const u32 sz = operands[0].size / 8;
    const u64 r = lhs ^ rhs;
    WriteOperand(s, inst, operands[0], r & OperandMask(sz * 8), false);
    UpdateFlagsLogic(s, r, sz * 8);
    return s.rip + inst.length;
}

HANDLER(Test) {
    const u64 lhs = ReadOperand(s, inst, operands[0], false);
    const u64 rhs = ReadOperand(s, inst, operands[1], false);
    const u32 sz = operands[0].size / 8;
    UpdateFlagsLogic(s, lhs & rhs, sz * 8);
    return s.rip + inst.length;
}

HANDLER(Cmp) {
    const u64 lhs = ReadOperand(s, inst, operands[0], false);
    const u64 rhs = ReadOperand(s, inst, operands[1], false);
    const u32 sz = operands[0].size / 8;
    UpdateFlagsSub(s, lhs, rhs, lhs - rhs, sz * 8);
    return s.rip + inst.length;
}

HANDLER(Inc) {
    const u64 lhs = ReadOperand(s, inst, operands[0], false);
    const u32 sz = operands[0].size / 8;
    const u64 r = (lhs + 1) & OperandMask(sz * 8);
    WriteOperand(s, inst, operands[0], r, false);
    UpdateFlagsInc(s, lhs, lhs + 1, sz * 8);
    return s.rip + inst.length;
}

HANDLER(Dec) {
    const u64 lhs = ReadOperand(s, inst, operands[0], false);
    const u32 sz = operands[0].size / 8;
    const u64 r = (lhs - 1) & OperandMask(sz * 8);
    WriteOperand(s, inst, operands[0], r, false);
    UpdateFlagsDec(s, lhs, lhs - 1, sz * 8);
    return s.rip + inst.length;
}

HANDLER(Not) {
    const u64 lhs = ReadOperand(s, inst, operands[0], false);
    const u32 sz = operands[0].size / 8;
    WriteOperand(s, inst, operands[0], ~lhs & OperandMask(sz * 8), false);
    // NOT does not affect flags.
    return s.rip + inst.length;
}

HANDLER(Neg) {
    const u64 lhs = ReadOperand(s, inst, operands[0], false);
    const u32 sz = operands[0].size / 8;
    const u64 r = (~lhs + 1) & OperandMask(sz * 8);
    WriteOperand(s, inst, operands[0], r, false);
    // NEG is equivalent to 0 - lhs.
    UpdateFlagsSub(s, 0, lhs, 0 - lhs, sz * 8);
    return s.rip + inst.length;
}

HANDLER(Jmp) {
    // For relative jumps, the immediate operand is a displacement
    // relative to the next instruction (RIP + length).
    const u64 raw = ReadOperand(s, inst, operands[0], true);
    if (operands[0].type == ZYDIS_OPERAND_TYPE_IMMEDIATE) {
        const s64 disp = static_cast<s64>(raw);
        return s.rip + inst.length + disp;
    }
    return raw; // register/indirect jump
}

HANDLER(Jcc) {
    if (EvaluateConditionCode(s, inst.mnemonic)) {
        const u64 raw = ReadOperand(s, inst, operands[0], true);
        if (operands[0].type == ZYDIS_OPERAND_TYPE_IMMEDIATE) {
            const s64 disp = static_cast<s64>(raw);
            return s.rip + inst.length + disp;
        }
        return raw;
    }
    return s.rip + inst.length;
}

HANDLER(Call) {
    // For relative calls, the immediate operand is a displacement
    // relative to the next instruction (RIP + length).
    const u64 raw = ReadOperand(s, inst, operands[0], true);
    u64 target;
    if (operands[0].type == ZYDIS_OPERAND_TYPE_IMMEDIATE) {
        const s64 disp = static_cast<s64>(raw);
        target = s.rip + inst.length + disp;
    } else {
        target = raw; // register/indirect call
    }
    const u64 ret = s.rip + inst.length;
    s.gpr[GPR_RSP] -= 8;
    WriteMemory(s.gpr[GPR_RSP], ret, 8, false);
    return target;
}

HANDLER(Ret) {
    // RET may have an immediate operand that gives the number of bytes
    // to pop after popping the return address (used by stdcall).  PS4
    // code uses near RET almost exclusively.
    const u64 ret = ReadMemory(s.gpr[GPR_RSP], 8, false);
    s.gpr[GPR_RSP] += 8;
    if (inst.operand_count_visible > 0 && operands[0].type == ZYDIS_OPERAND_TYPE_IMMEDIATE) {
        s.gpr[GPR_RSP] += static_cast<u64>(static_cast<s64>(operands[0].imm.value.s));
    }
    return ret;
}

HANDLER(Nop) {
    return s.rip + inst.length;
}

HANDLER(Xchg) {
    const u64 a = ReadOperand(s, inst, operands[0], false);
    const u64 b = ReadOperand(s, inst, operands[1], false);
    WriteOperand(s, inst, operands[0], b, false);
    WriteOperand(s, inst, operands[1], a, false);
    return s.rip + inst.length;
}

HANDLER(Movsx) {
    const u64 src = ReadOperand(s, inst, operands[1], true);
    const u32 src_bits = operands[1].size;
    const u64 sign_extended = [src, src_bits]() -> u64 {
        switch (src_bits) {
        case 8:  return static_cast<u64>(static_cast<s8>(src));
        case 16: return static_cast<u64>(static_cast<s16>(src));
        case 32: return static_cast<u64>(static_cast<s32>(src));
        default: return src;
        }
    }();
    WriteOperand(s, inst, operands[0], sign_extended, false);
    return s.rip + inst.length;
}

HANDLER(Movzx) {
    const u64 src = ReadOperand(s, inst, operands[1], false);
    WriteOperand(s, inst, operands[0], src, false);
    return s.rip + inst.length;
}

HANDLER(Cmov) {
    // Strip the CMOV-conditional family to its Jcc equivalent by mapping
    // CMOVO -> JO etc. We use EvaluateConditionCode with the CMOV's own
    // mnemonic which doesn't match — we need the Jcc mnemonic. The
    // simplest way is to evaluate via the CMOV family directly: look
    // up Jcc by offset.
    static const struct {
        ZydisMnemonic_ cmov;
        ZydisMnemonic_ jcc;
    } kMap[] = {
        {ZYDIS_MNEMONIC_CMOVO,  ZYDIS_MNEMONIC_JO},
        {ZYDIS_MNEMONIC_CMOVNO, ZYDIS_MNEMONIC_JNO},
        {ZYDIS_MNEMONIC_CMOVB,  ZYDIS_MNEMONIC_JB},
        {ZYDIS_MNEMONIC_CMOVNB, ZYDIS_MNEMONIC_JNB},
        {ZYDIS_MNEMONIC_CMOVZ,  ZYDIS_MNEMONIC_JZ},
        {ZYDIS_MNEMONIC_CMOVNZ, ZYDIS_MNEMONIC_JNZ},
        {ZYDIS_MNEMONIC_CMOVBE, ZYDIS_MNEMONIC_JBE},
        {ZYDIS_MNEMONIC_CMOVNBE,ZYDIS_MNEMONIC_JNBE},
        {ZYDIS_MNEMONIC_CMOVS,  ZYDIS_MNEMONIC_JS},
        {ZYDIS_MNEMONIC_CMOVNS, ZYDIS_MNEMONIC_JNS},
        {ZYDIS_MNEMONIC_CMOVP,  ZYDIS_MNEMONIC_JP},
        {ZYDIS_MNEMONIC_CMOVNP, ZYDIS_MNEMONIC_JNP},
        {ZYDIS_MNEMONIC_CMOVL,  ZYDIS_MNEMONIC_JL},
        {ZYDIS_MNEMONIC_CMOVNL, ZYDIS_MNEMONIC_JNL},
        {ZYDIS_MNEMONIC_CMOVLE, ZYDIS_MNEMONIC_JLE},
        {ZYDIS_MNEMONIC_CMOVNLE,ZYDIS_MNEMONIC_JNLE},
    };
    for (const auto& e : kMap) {
        if (e.cmov == inst.mnemonic) {
            if (EvaluateConditionCode(s, e.jcc)) {
                const u64 v = ReadOperand(s, inst, operands[1], false);
                WriteOperand(s, inst, operands[0], v, false);
            }
            return s.rip + inst.length;
        }
    }
    return s.rip + inst.length;
}

HANDLER(Setcc) {
    // SETcc stores 1 or 0 in its destination based on the condition.
    // We map SETcc -> Jcc.
    static const struct {
        ZydisMnemonic_ setcc;
        ZydisMnemonic_ jcc;
    } kMap[] = {
        {ZYDIS_MNEMONIC_SETO,  ZYDIS_MNEMONIC_JO},
        {ZYDIS_MNEMONIC_SETNO, ZYDIS_MNEMONIC_JNO},
        {ZYDIS_MNEMONIC_SETB,  ZYDIS_MNEMONIC_JB},
        {ZYDIS_MNEMONIC_SETNB, ZYDIS_MNEMONIC_JNB},
        {ZYDIS_MNEMONIC_SETZ,  ZYDIS_MNEMONIC_JZ},
        {ZYDIS_MNEMONIC_SETNZ, ZYDIS_MNEMONIC_JNZ},
        {ZYDIS_MNEMONIC_SETBE, ZYDIS_MNEMONIC_JBE},
        {ZYDIS_MNEMONIC_SETNBE,ZYDIS_MNEMONIC_JNBE},
        {ZYDIS_MNEMONIC_SETS,  ZYDIS_MNEMONIC_JS},
        {ZYDIS_MNEMONIC_SETNS, ZYDIS_MNEMONIC_JNS},
        {ZYDIS_MNEMONIC_SETP,  ZYDIS_MNEMONIC_JP},
        {ZYDIS_MNEMONIC_SETNP, ZYDIS_MNEMONIC_JNP},
        {ZYDIS_MNEMONIC_SETL,  ZYDIS_MNEMONIC_JL},
        {ZYDIS_MNEMONIC_SETNL, ZYDIS_MNEMONIC_JNL},
        {ZYDIS_MNEMONIC_SETLE, ZYDIS_MNEMONIC_JLE},
        {ZYDIS_MNEMONIC_SETNLE,ZYDIS_MNEMONIC_JNLE},
    };
    for (const auto& e : kMap) {
        if (e.setcc == inst.mnemonic) {
            const bool taken = EvaluateConditionCode(s, e.jcc);
            WriteOperand(s, inst, operands[0], taken ? 1 : 0, false);
            return s.rip + inst.length;
        }
    }
    return s.rip + inst.length;
}

HANDLER(Shl) {
    const u64 lhs = ReadOperand(s, inst, operands[0], false);
    const u64 rhs = ReadOperand(s, inst, operands[1], false) & 0x3F; // masked to 6 bits in 64-bit mode
    const u32 sz = operands[0].size;
    const u64 mask = OperandMask(sz);
    const u64 r = (lhs << rhs) & mask;
    WriteOperand(s, inst, operands[0], r, false);
    // Basic flag update — CF gets the last bit shifted out, ZF/SF/PF
    // updated from the result, OF is only defined for count=1.
    if (rhs != 0) {
        SetFlag(s, RflagsBits::CF, (lhs >> (sz - rhs)) & 1);
    }
    UpdateFlagsLogic(s, r, sz); // sets ZF/SF/PF, clears CF/OF — we overwrite CF above
    return s.rip + inst.length;
}

HANDLER(Shr) {
    const u64 lhs = ReadOperand(s, inst, operands[0], false);
    const u64 rhs = ReadOperand(s, inst, operands[1], false) & 0x3F;
    const u32 sz = operands[0].size;
    const u64 mask = OperandMask(sz);
    const u64 r = (lhs >> rhs) & mask;
    WriteOperand(s, inst, operands[0], r, false);
    if (rhs != 0) {
        SetFlag(s, RflagsBits::CF, (lhs >> (rhs - 1)) & 1);
    }
    UpdateFlagsLogic(s, r, sz);
    return s.rip + inst.length;
}

HANDLER(Sar) {
    const u64 lhs = ReadOperand(s, inst, operands[0], false);
    const u64 rhs = ReadOperand(s, inst, operands[1], false) & 0x3F;
    const u32 sz = operands[0].size;
    const u64 mask = OperandMask(sz);
    // Sign-extend lhs first.
    const s64 sval = (sz == 64) ? static_cast<s64>(lhs)
                   : (sz == 32) ? static_cast<s64>(static_cast<s32>(lhs & 0xFFFF'FFFFULL))
                   : (sz == 16) ? static_cast<s64>(static_cast<s16>(lhs & 0xFFFFULL))
                   :             static_cast<s64>(static_cast<s8>(lhs & 0xFFULL));
    const u64 r = static_cast<u64>(sval >> rhs) & mask;
    WriteOperand(s, inst, operands[0], r, false);
    if (rhs != 0) {
        SetFlag(s, RflagsBits::CF, (lhs >> (rhs - 1)) & 1);
    }
    UpdateFlagsLogic(s, r, sz);
    return s.rip + inst.length;
}

HANDLER(Mul) {
    // One-operand MUL: RDX:RAX = RAX * src. Only sets CF/OF (cleared if
    // the high half is zero, set otherwise).
    const u64 src = ReadOperand(s, inst, operands[0], false);
    const u32 sz = operands[0].size;
    const u64 a = GetRegValue(s, ZYDIS_REGISTER_RAX, sz / 8);
    if (sz == 8) {
        // 64-bit MUL: result is 128 bits. Use __uint128_t for the
        // high-half capture.
        const __uint128_t product = static_cast<__uint128_t>(a) * static_cast<__uint128_t>(src);
        SetRegValue(s, ZYDIS_REGISTER_RAX, static_cast<u64>(product), 8);
        SetRegValue(s, ZYDIS_REGISTER_RDX, static_cast<u64>(product >> 64), 8);
        SetFlag(s, RflagsBits::CF, (product >> 64) != 0);
        SetFlag(s, RflagsBits::OF, (product >> 64) != 0);
    } else if (sz == 4) {
        const u64 product = a * src;
        SetRegValue(s, ZYDIS_REGISTER_RAX, product & 0xFFFF'FFFFULL, 4);
        SetRegValue(s, ZYDIS_REGISTER_RDX, (product >> 32) & 0xFFFF'FFFFULL, 4);
        SetFlag(s, RflagsBits::CF, (product >> 32) != 0);
        SetFlag(s, RflagsBits::OF, (product >> 32) != 0);
    } else if (sz == 2) {
        const u32 product = static_cast<u32>(a * src);
        SetRegValue(s, ZYDIS_REGISTER_RAX, product & 0xFFFF, 2);
        SetRegValue(s, ZYDIS_REGISTER_RDX, (product >> 16) & 0xFFFF, 2);
        SetFlag(s, RflagsBits::CF, (product >> 16) != 0);
        SetFlag(s, RflagsBits::OF, (product >> 16) != 0);
    } else { // 1 byte
        const u16 product = static_cast<u16>(static_cast<u8>(a) * static_cast<u8>(src));
        SetRegValue(s, ZYDIS_REGISTER_RAX, product, 2);
        SetFlag(s, RflagsBits::CF, (product >> 8) != 0);
        SetFlag(s, RflagsBits::OF, (product >> 8) != 0);
    }
    return s.rip + inst.length;
}

HANDLER(Imul) {
    // Three forms: 1-operand (RDX:RAX = RAX * src), 2-operand (dst *= src),
    // 3-operand (dst = src1 * src2).
    if (inst.operand_count_visible == 1) {
        const s64 src = static_cast<s64>(ReadOperand(s, inst, operands[0], true));
        const u32 sz = operands[0].size;
        const s64 a = static_cast<s64>(GetRegValue(s, ZYDIS_REGISTER_RAX, sz / 8));
        const __int128 product = static_cast<__int128>(a) * static_cast<__int128>(src);
        if (sz == 8) {
            SetRegValue(s, ZYDIS_REGISTER_RAX, static_cast<u64>(product), 8);
            SetRegValue(s, ZYDIS_REGISTER_RDX, static_cast<u64>(product >> 64), 8);
        } else if (sz == 4) {
            const s64 product_64 = static_cast<s64>(static_cast<s32>(a) * static_cast<s32>(src));
            SetRegValue(s, ZYDIS_REGISTER_RAX, static_cast<u64>(product_64) & 0xFFFF'FFFFULL, 4);
            SetRegValue(s, ZYDIS_REGISTER_RDX, (static_cast<u64>(product_64) >> 32) & 0xFFFF'FFFFULL, 4);
        } else if (sz == 2) {
            const s32 product_32 = static_cast<s32>(static_cast<s16>(a) * static_cast<s16>(src));
            SetRegValue(s, ZYDIS_REGISTER_RAX, static_cast<u32>(product_32) & 0xFFFF, 2);
            SetRegValue(s, ZYDIS_REGISTER_RDX, (static_cast<u32>(product_32) >> 16) & 0xFFFF, 2);
        } else {
            const s16 product_16 = static_cast<s16>(static_cast<s8>(a) * static_cast<s8>(src));
            SetRegValue(s, ZYDIS_REGISTER_RAX, static_cast<u16>(product_16), 2);
        }
    } else if (inst.operand_count_visible == 2) {
        const u32 sz = operands[0].size;
        const s64 lhs = static_cast<s64>(ReadOperand(s, inst, operands[0], true));
        const s64 rhs = static_cast<s64>(ReadOperand(s, inst, operands[1], true));
        const __int128 product = static_cast<__int128>(lhs) * static_cast<__int128>(rhs);
        const u64 result = static_cast<u64>(product & (sz == 8 ? ~0ULL : ((1ULL << sz) - 1)));
        WriteOperand(s, inst, operands[0], result, false);
        // Overflow flag: set if the result doesn't fit in the operand size.
        const __int128 low = static_cast<__int128>(static_cast<s64>(result));
        bool overflow = (product != static_cast<__int128>(low));
        if (sz < 64) {
            // sign-extend result back to __int128 and compare
            const s64 sext = (sz == 32) ? static_cast<s64>(static_cast<s32>(result & 0xFFFF'FFFFULL))
                           : (sz == 16) ? static_cast<s64>(static_cast<s16>(result & 0xFFFFULL))
                           :              static_cast<s64>(static_cast<s8>(result & 0xFFULL));
            overflow = (static_cast<__int128>(sext) != product);
        }
        SetFlag(s, RflagsBits::CF, overflow);
        SetFlag(s, RflagsBits::OF, overflow);
    } else { // 3 operands
        const u32 sz = operands[0].size;
        const s64 src1 = static_cast<s64>(ReadOperand(s, inst, operands[1], true));
        const s64 src2 = static_cast<s64>(ReadOperand(s, inst, operands[2], true));
        const __int128 product = static_cast<__int128>(src1) * static_cast<__int128>(src2);
        const u64 result = static_cast<u64>(product & (sz == 8 ? ~0ULL : ((1ULL << sz) - 1)));
        WriteOperand(s, inst, operands[0], result, false);
        bool overflow;
        if (sz == 8) {
            overflow = (static_cast<s64>(result) != product);
        } else if (sz == 4) {
            overflow = (static_cast<s64>(static_cast<s32>(result & 0xFFFF'FFFFULL)) != product);
        } else if (sz == 2) {
            overflow = (static_cast<s64>(static_cast<s16>(result & 0xFFFFULL)) != product);
        } else {
            overflow = (static_cast<s64>(static_cast<s8>(result & 0xFFULL)) != product);
        }
        SetFlag(s, RflagsBits::CF, overflow);
        SetFlag(s, RflagsBits::OF, overflow);
    }
    return s.rip + inst.length;
}

HANDLER(Div) {
    // Unsigned DIV: RDX:RAX / src -> RAX = quotient, RDX = remainder.
    const u64 src = ReadOperand(s, inst, operands[0], false);
    const u32 sz = operands[0].size;
    if (src == 0) {
        LOG_ERROR(Core_Cpu, "DIV by zero at rip=0x{:016x}", s.rip);
        return s.rip + inst.length; // would trap on real hardware
    }
    if (sz == 8) {
        const __uint128_t dividend = (static_cast<__uint128_t>(GetRegValue(s, ZYDIS_REGISTER_RDX, 8)) << 64) |
                              GetRegValue(s, ZYDIS_REGISTER_RAX, 8);
        SetRegValue(s, ZYDIS_REGISTER_RAX, static_cast<u64>(dividend / src), 8);
        SetRegValue(s, ZYDIS_REGISTER_RDX, static_cast<u64>(dividend % src), 8);
    } else if (sz == 4) {
        const u64 dividend = (GetRegValue(s, ZYDIS_REGISTER_RDX, 4) << 32) |
                             GetRegValue(s, ZYDIS_REGISTER_RAX, 4);
        SetRegValue(s, ZYDIS_REGISTER_RAX, dividend / src, 4);
        SetRegValue(s, ZYDIS_REGISTER_RDX, dividend % src, 4);
    } else if (sz == 2) {
        const u32 dividend = (GetRegValue(s, ZYDIS_REGISTER_RDX, 2) << 16) |
                             GetRegValue(s, ZYDIS_REGISTER_RAX, 2);
        SetRegValue(s, ZYDIS_REGISTER_RAX, dividend / src, 2);
        SetRegValue(s, ZYDIS_REGISTER_RDX, dividend % src, 2);
    } else {
        const u16 dividend = static_cast<u16>(GetRegValue(s, ZYDIS_REGISTER_RAX, 2));
        SetRegValue(s, ZYDIS_REGISTER_RAX, (dividend / static_cast<u8>(src)) & 0xFF, 1);
        SetRegValue(s, ZYDIS_REGISTER_RAX, (dividend % static_cast<u8>(src)) << 8, 1);
    }
    return s.rip + inst.length;
}

HANDLER(Idiv) {
    // Signed IDIV: RDX:RAX / src -> RAX = quotient, RDX = remainder.
    const s64 src = static_cast<s64>(ReadOperand(s, inst, operands[0], true));
    const u32 sz = operands[0].size;
    if (src == 0) {
        LOG_ERROR(Core_Cpu, "IDIV by zero at rip=0x{:016x}", s.rip);
        return s.rip + inst.length;
    }
    if (sz == 8) {
        const __int128 dividend = (static_cast<__int128>(static_cast<s64>(GetRegValue(s, ZYDIS_REGISTER_RDX, 8))) << 64) |
                              static_cast<__int128>(static_cast<s64>(GetRegValue(s, ZYDIS_REGISTER_RAX, 8)));
        SetRegValue(s, ZYDIS_REGISTER_RAX, static_cast<u64>(static_cast<s64>(dividend / src)), 8);
        SetRegValue(s, ZYDIS_REGISTER_RDX, static_cast<u64>(static_cast<s64>(dividend % src)), 8);
    } else if (sz == 4) {
        const s64 dividend = (static_cast<s64>(static_cast<s32>(GetRegValue(s, ZYDIS_REGISTER_RDX, 4))) << 32) |
                              static_cast<s64>(static_cast<s32>(GetRegValue(s, ZYDIS_REGISTER_RAX, 4)));
        SetRegValue(s, ZYDIS_REGISTER_RAX, static_cast<u64>(static_cast<s32>(dividend / src)), 4);
        SetRegValue(s, ZYDIS_REGISTER_RDX, static_cast<u64>(static_cast<s32>(dividend % src)), 4);
    } else if (sz == 2) {
        const s32 dividend = (static_cast<s32>(static_cast<s16>(GetRegValue(s, ZYDIS_REGISTER_RDX, 2))) << 16) |
                              static_cast<s32>(static_cast<s16>(GetRegValue(s, ZYDIS_REGISTER_RAX, 2)));
        SetRegValue(s, ZYDIS_REGISTER_RAX, static_cast<u16>(static_cast<s16>(dividend / src)), 2);
        SetRegValue(s, ZYDIS_REGISTER_RDX, static_cast<u16>(static_cast<s16>(dividend % src)), 2);
    } else {
        const s16 dividend = static_cast<s16>(GetRegValue(s, ZYDIS_REGISTER_RAX, 2));
        SetRegValue(s, ZYDIS_REGISTER_RAX, static_cast<u8>(static_cast<s8>(dividend / static_cast<s8>(src))), 1);
        SetRegValue(s, ZYDIS_REGISTER_RAX, (static_cast<u16>(static_cast<s8>(dividend % static_cast<s8>(src))) << 8), 1);
    }
    return s.rip + inst.length;
}

HANDLER(Xadd) {
    // XADD dst, src: temp = dst; dst = dst + src; src = temp.
    const u64 a = ReadOperand(s, inst, operands[0], false);
    const u64 b = ReadOperand(s, inst, operands[1], false);
    const u32 sz = operands[0].size / 8;
    const u64 sum = (a + b) & OperandMask(sz * 8);
    WriteOperand(s, inst, operands[0], sum, false);
    WriteOperand(s, inst, operands[1], a, false);
    UpdateFlagsAdd(s, a, b, a + b, sz * 8);
    return s.rip + inst.length;
}

HANDLER(CmpXchg) {
    // CMPXCHG dst, src: if RAX == dst then dst = src else RAX = dst.
    const u32 sz = operands[0].size / 8;
    const u64 acc = GetRegValue(s, ZYDIS_REGISTER_RAX, sz);
    const u64 dst = ReadOperand(s, inst, operands[0], false);
    const u64 src = ReadOperand(s, inst, operands[1], false);
    if (acc == dst) {
        WriteOperand(s, inst, operands[0], src, false);
        SetFlag(s, RflagsBits::ZF, true);
    } else {
        SetRegValue(s, ZYDIS_REGISTER_RAX, dst, sz);
        SetFlag(s, RflagsBits::ZF, false);
    }
    // Other flags updated as if SUB had run.
    UpdateFlagsSub(s, acc, dst, acc - dst, sz * 8);
    return s.rip + inst.length;
}

// ─────────────────────────────────────────────────────────────────────────
//  Main loop
// ─────────────────────────────────────────────────────────────────────────

u64 X64InterpreterBackend::Execute(u64 rip, const GuestCallContext& /*ctx*/) {
    X64CpuState state{};
    state.rip = rip;
    state.in_guest_code = true;

    // Android/ARM64 port: allocate a guest stack for the interpreter.
    // On x86 hosts, Linker::RunMainEntry sets up the stack via inline
    // assembly (push entry params, align RSP, etc.). On ARM64 we skip
    // RunMainEntry entirely and go straight to the interpreter, so we
    // need to set up the stack ourselves.
    //
    // We mmap a 1 MB stack at a fixed guest address inside the user
    // region. RSP starts at the top (stack grows downward). We also
    // push a sentinel return address (0) so the first RET exits the
    // interpreter loop (via the HLE bridge or by returning to address
    // 0 which will fail to decode and break the loop).
#if defined(__ANDROID__)
    {
        constexpr u64 kStackSize = 1 * 1024 * 1024; // 1 MB
        constexpr u64 kStackBase = 0x3A0000000ULL;   // inside user region
        void* stack_ptr = mmap(reinterpret_cast<void*>(kStackBase), kStackSize,
                               PROT_READ | PROT_WRITE,
                               MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0);
        if (stack_ptr == MAP_FAILED) {
            LOG_CRITICAL(Core_Cpu, "Failed to allocate guest stack: {}", strerror(errno));
            return rip;
        }
        // Stack grows downward; RSP starts at the top.
        state.gpr[GPR_RSP] = kStackBase + kStackSize - 16;
        // 16-byte align (PS4 ABI requires 16-byte aligned RSP at entry,
        // but videoout_basic expects it misaligned by 8, so we leave
        // RSP at (top - 16) which is 16-aligned).
        // Push a sentinel return address (0) so RET exits.
        state.gpr[GPR_RSP] -= 8;
        WriteMemory(state.gpr[GPR_RSP], 0, 8, false);
        // Misalign by 8 (videoout_basic expects this).
        state.gpr[GPR_RSP] -= 8;
        WriteMemory(state.gpr[GPR_RSP], 0, 8, false);
        LOG_INFO(Core_Cpu, "Guest stack allocated: 0x{:x} - 0x{:x}, RSP=0x{:x}",
                  kStackBase, kStackBase + kStackSize, state.gpr[GPR_RSP]);
    }
#endif

    ZydisDecodedInstruction inst{};
    ZydisDecodedOperand operands[ZYDIS_MAX_OPERAND_COUNT];

    u64 instruction_count = 0;
    while (state.in_guest_code) {
        // Sanity check: if RIP is outside the valid guest memory range,
        // it's a bad jump/return from an instruction we haven't verified.
        // Guest memory is at 0x200000000 - 0x900000000. Any RIP outside
        // this range would cause a SIGSEGV when we try to read guest
        // code, which would kill the process on Android (the signal
        // handler calls raise(SIGSEGV) to terminate). Instead, log
        // and exit the interpreter loop gracefully.
        if (state.rip < 0x100000000ULL || state.rip >= 0x900000000ULL) {
            LOG_ERROR(Core_Cpu, "Interpreter: RIP=0x{:x} is outside guest memory range "
                      "(0x200000000-0x900000000). Likely bad jump/ret from "
                      "unimplemented instruction. Exiting gracefully.",
                      state.rip);
            DumpRecentInterpreterTrace("bad-rip", 32);
            state.in_guest_code = false;
            break;
        }

        // Optionally consult the JIT's native block provider — if it
        // returns a valid block we'd dispatch to it instead of decoding
        // here. For now this is a no-op (the JIT isn't wired up yet).
        if (m_impl->native_block_provider) {
            const NativeBlockLookup blk = m_impl->native_block_provider(state.rip);
            (void)blk; // suppress unused warning until JIT lands
        }

        // Decode one instruction at the current RIP. RIP is a guest
        // virtual address that's mapped 1:1 into the host process, so
        // we can just reinterpret_cast it to a host pointer for the
        // decode. We copy up to 15 bytes into a local buffer first to
        // avoid Zydis directly dereferencing the guest pointer (which
        // may trigger a signal if the guest page isn't fully mapped).
        const u8* code = reinterpret_cast<const u8*>(state.rip);
        u8 code_buf[15];
        // Copy guest bytes into local buffer FIRST, before any logging
        // that might dereference the pointer. If the page is unmapped,
        // the memcpy will trigger SIGSEGV — but we've already checked
        // the RIP range above, so this should be safe.
        std::memcpy(code_buf, code, 15);
        // Log first few instructions for debugging.
        if (instruction_count < 10) {
            LOG_INFO(Core_Cpu, "[{:3}] rip=0x{:016x} rsp=0x{:016x} bytes={:02x} {:02x} {:02x} {:02x}",
                      instruction_count, state.rip, state.gpr[GPR_RSP],
                      code_buf[0], code_buf[1], code_buf[2], code_buf[3]);
        }
        const ZyanStatus status = ZydisDecoderDecodeFull(
            &m_impl->decoder, code_buf, 15, &inst, operands);
        if (!ZYAN_SUCCESS(status)) {
            LOG_ERROR(Core_Cpu, "Decoder failed at rip=0x{:016x} (status=0x{:08x})",
                      state.rip, status);
            DumpRecentInterpreterTrace("decoder-fail", 32);
            state.in_guest_code = false;
            break;
        }

        if (m_impl->config.trace) {
            LOG_TRACE(Core_Cpu, "[{:4}] rip=0x{:016x} mnem={}",
                      instruction_count, state.rip, static_cast<unsigned>(inst.mnemonic));
        }
        RecordTrace(state.rip, inst.mnemonic);

        // HLE bridge: if the current RIP points into the AeroLib stub
        // arena, we got here via a CALL/JMP that the linker patched to
        // a stub address. Divert to DispatchAerolibStub which calls
        // the HLE handler directly and updates RIP to the guest's
        // return address.
        if (IsAerolibStubAddress(state.rip)) {
            if (DispatchAerolibStub(state.rip, &state)) {
                ++instruction_count;
                if (m_impl->config.max_instructions != 0 &&
                    instruction_count >= m_impl->config.max_instructions) {
                    LOG_WARNING(Core_Cpu, "Interpreter hit instruction cap {} — exiting",
                                m_impl->config.max_instructions);
                    break;
                }
                continue;
            }
        }

        // Try SSE / XMM dispatcher first — if it handled the mnemonic
        // we treat the instruction as executed and advance RIP by
        // inst.length.
        bool handled = HandleSseInstruction(state, inst, operands);
        if (!handled) handled = HandleShiftBitInstruction(state, inst, operands);
        if (!handled) handled = HandleStringRepInstruction(state, inst, operands);
        if (!handled) handled = HandleSystemLockInstruction(state, inst, operands);
        if (handled) {
            state.rip += inst.length;
            ++instruction_count;
            if (m_impl->config.max_instructions != 0 &&
                instruction_count >= m_impl->config.max_instructions) {
                LOG_WARNING(Core_Cpu, "Interpreter hit instruction cap {} — exiting",
                            m_impl->config.max_instructions);
                break;
            }
            continue;
        }

        Handler h = LookupHandler(inst.mnemonic);
        if (h == nullptr) {
            LOG_ERROR(Core_Cpu, "Unimplemented x86-64 mnemonic {} (0x{:x}) at rip=0x{:016x}",
                      static_cast<unsigned>(inst.mnemonic), static_cast<unsigned>(inst.mnemonic),
                      state.rip);
            if (m_impl->config.abort_on_unknown) {
                DumpRecentInterpreterTrace("unimplemented-mnemonic", 32);
                state.in_guest_code = false;
                break;
            }
            // Skip the unknown instruction and continue — many PS4
            // homebrews tolerate a skipped SSE/AVX if the result is
            // never read. (Long-term we'll implement them all.)
            state.rip += inst.length;
        } else {
            state.rip = h(state, inst, operands);
        }

        ++instruction_count;
        if (m_impl->config.max_instructions != 0 &&
            instruction_count >= m_impl->config.max_instructions) {
            LOG_WARNING(Core_Cpu, "Interpreter hit instruction cap {} — exiting",
                        m_impl->config.max_instructions);
            break;
        }
    }

    return state.rip;
}

// ─────────────────────────────────────────────────────────────────────────
//  ExecuteOneInstruction — runs exactly one x86-64 instruction using
//  the caller-provided X64CpuState. Used by the JIT for single-instruction
//  fallback when it encounters an instruction it can't translate.
// ─────────────────────────────────────────────────────────────────────────

u64 X64InterpreterBackend::ExecuteOneInstruction(X64CpuState& state) {
    // Validate RIP.
    if (state.rip < 0x100000000ULL || state.rip >= 0x900000000ULL) {
        LOG_ERROR(Core_Cpu, "ExecuteOneInstruction: RIP=0x{:x} outside guest memory", state.rip);
        return state.rip;
    }

    // Read up to 15 bytes of guest code at state.rip.
    const u8* code = reinterpret_cast<const u8*>(state.rip);
    u8 code_buf[15];
    std::memcpy(code_buf, code, 15);

    // Decode the instruction.
    ZydisDecodedInstruction inst{};
    ZydisDecodedOperand operands[ZYDIS_MAX_OPERAND_COUNT];

    const ZyanStatus status = ZydisDecoderDecodeFull(
        &m_impl->decoder, code_buf, 15, &inst, operands);
    if (!ZYAN_SUCCESS(status)) {
        LOG_ERROR(Core_Cpu, "ExecuteOneInstruction: decode failed at rip=0x{:x} (0x{:x})",
                  state.rip, status);
        state.rip += 1; // skip one byte
        return state.rip;
    }

    // Check for HLE stub.
    if (IsAerolibStubAddress(state.rip)) {
        if (DispatchAerolibStub(state.rip, &state)) {
            return state.rip;
        }
    }

    // Try SSE / XMM / shift / string / system handlers.
    bool handled = HandleSseInstruction(state, inst, operands);
    if (!handled) handled = HandleShiftBitInstruction(state, inst, operands);
    if (!handled) handled = HandleStringRepInstruction(state, inst, operands);
    if (!handled) handled = HandleSystemLockInstruction(state, inst, operands);
    if (handled) {
        state.rip += inst.length;
        return state.rip;
    }

    // Try the main handler table.
    Handler h = LookupHandler(inst.mnemonic);
    if (h == nullptr) {
        LOG_ERROR(Core_Cpu, "ExecuteOneInstruction: unimplemented mnemonic {} at rip=0x{:x}",
                  static_cast<unsigned>(inst.mnemonic), state.rip);
        state.rip += inst.length; // skip
        return state.rip;
    }
    state.rip = h(state, inst, operands);
    return state.rip;
}

} // namespace Core::Cpu
