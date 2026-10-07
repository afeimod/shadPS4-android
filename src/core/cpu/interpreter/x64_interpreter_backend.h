// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later
//
// X64InterpreterBackend — Zydis-driven instruction-by-instruction
// interpreter for x86-64 guest code on ARM64 (or any non-x86 host).
//
// Execution model:
//   Execute(rip, ctx) sets up an X64CpuState, maps the guest `rip` to a
//   host pointer, then enters the main loop:
//     1. Read bytes at guest RIP into a small buffer (16 bytes is enough
//        for any single x86-64 instruction).
//     2. Use ZydisDecoderDecodeInstruction to disassemble them.
//     3. Dispatch on the mnemonic to a per-instruction handler that
//        reads the operands via ReadOperand(), mutates the CPU state,
//        and advances RIP by instruction.length.
//     4. Repeat until a non-local jump (HLT, ret-to-host, page fault we
//        can't service, HLE call) is requested.
//
// This is slow (10-100× slower than native) but correct; once the
// Aarch64JitBackend lands we still need it for cold paths and for
// validating JIT output.

#pragma once

#include <atomic>
#include <functional>
#include <memory>

#include <Zydis/Zydis.h>

#include "core/cpu/cpu_backend.h"
#include "core/cpu/x64_cpu_state.h"

namespace Core::Cpu {

// A lookup that the JIT can use to short-circuit a guest RIP to an
// already-translated native block. The interpreter ignores this for now
// but exposes the hook so the JIT can register itself when wired up.
struct NativeBlockLookup {
    void* host_code = nullptr;       // Pointer to translated ARM64 code
    u64   host_code_size = 0;        // Size in bytes
    u64   guest_size = 0;            // How many guest bytes the block covers
    bool  valid = false;             // Whether this entry is populated
};

// Runtime config knobs for the interpreter — passed in via
// SetRuntimeConfig with a string key/value pair.
struct X64InterpreterRuntimeConfig {
    // When true, the interpreter logs every guest instruction it
    // executes (verbose, only for debugging specific games).
    bool trace = false;
    // When true, the interpreter stops after `max_instructions` (set to 0
    // for unbounded). Useful for catching infinite loops during dev.
    u64  max_instructions = 0;
    // When true, the interpreter aborts on the first unknown mnemonic
    // instead of skipping it. Default false (skip with a log line).
    bool abort_on_unknown = false;
};

class X64InterpreterBackend : public CpuBackend {
public:
    X64InterpreterBackend();
    ~X64InterpreterBackend() override;

    // CpuBackend interface
    u64 Execute(u64 rip, const GuestCallContext& ctx) override;
    CpuBackendKind Kind() const override { return CpuBackendKind::X64Interpreter; }
    void SetRuntimeConfig(const X64InterpreterRuntimeConfig& cfg);

    // Lets the JIT (when wired up) register a callback the interpreter
    // can consult before falling back to the slow path. Default impl
    // returns an invalid lookup so the interpreter always decodes.
    void SetNativeBlockProvider(
        std::function<NativeBlockLookup(u64)> provider);

    // Public for diagnostic dump tools (debugger, devtools). The internal
    // state is per-thread, so these only inspect the calling thread's
    // current guest state.
    static void SetInterpreterThreadStackHint(u64 guest_rip, u64 stack_top);

    // Execute a single x86-64 instruction using the provided state.
    // Decodes the instruction at state.rip, dispatches it to the
    // appropriate handler, updates state (GPRs, RIP, flags), and
    // returns the new RIP. This is used by the JIT backend for
    // single-instruction fallback — the JIT passes its own
    // X64CpuState so GPRs stay synchronized.
    u64 ExecuteOneInstruction(X64CpuState& state);

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

// Free helpers exposed for the JIT and the diagnostic dump tools.
u64  GetRegValue(const X64CpuState& state, ZydisRegister reg, u32 operand_size);
void SetRegValue(X64CpuState& state, ZydisRegister reg, u64 value, u32 operand_size);

u64  ReadMemory(u64 guest_addr, u32 size, bool is_signed);
void WriteMemory(u64 guest_addr, u64 value, u32 size, bool is_signed);

u64  ReadOperand(const X64CpuState& state,
                 const ZydisDecodedInstruction& inst,
                 const ZydisDecodedOperand& op,
                 bool is_signed);
void WriteOperand(X64CpuState& state,
                 const ZydisDecodedInstruction& inst,
                 const ZydisDecodedOperand& op,
                 u64 value,
                 bool is_signed);

// Flag-update helpers. The interpreter calls these after each arithmetic
// / logical / increment / decrement / adc / sbb to keep RFLAGS in sync.
// `size` is the operand size in bits (8/16/32/64); `result` and `lhs` are
// zero-extended to u64.
void UpdateFlagsAdd(X64CpuState& s, u64 lhs, u64 rhs, u64 result, u32 size);
void UpdateFlagsSub(X64CpuState& s, u64 lhs, u64 rhs, u64 result, u32 size);
void UpdateFlagsAdc(X64CpuState& s, u64 lhs, u64 rhs, bool carry_in, u64 result, u32 size);
void UpdateFlagsSbb(X64CpuState& s, u64 lhs, u64 rhs, bool carry_in, u64 result, u32 size);
void UpdateFlagsLogic(X64CpuState& s, u64 result, u32 size);
void UpdateFlagsInc(X64CpuState& s, u64 lhs, u64 result, u32 size);
void UpdateFlagsDec(X64CpuState& s, u64 lhs, u64 result, u32 size);

// Evaluate a Jcc / CMOVcc / SETcc condition code against the current
// RFLAGS.  Returns true if the condition is met.
bool EvaluateConditionCode(const X64CpuState& s, ZydisMnemonic mnemonic);

// Diagnostics: dump the last N interpreter steps to the log.
void DumpRecentInterpreterTrace(const char* tag, unsigned int count);

// SSE / XMM handler dispatch table (defined in x64_sse_handlers.cpp).
// Returns true if the mnemonic was handled, false if it needs the
// integer dispatcher to try.
bool HandleSseInstruction(X64CpuState& s,
                          const ZydisDecodedInstruction& inst,
                          const ZydisDecodedOperand* operands);

// Shift / rotate / bit-test / bit-scan dispatcher (defined in
// x64_shift_string_handlers.cpp).
bool HandleShiftBitInstruction(X64CpuState& s,
                               const ZydisDecodedInstruction& inst,
                               const ZydisDecodedOperand* operands);

// String ops + REP prefix dispatcher (same TU as shift/bit).
bool HandleStringRepInstruction(X64CpuState& s,
                                const ZydisDecodedInstruction& inst,
                                const ZydisDecodedOperand* operands);

// System / synchronization / control-flow-misc dispatcher (defined in
// x64_system_lock_handlers.cpp). Covers SYSCALL / INT3 / UD2 / HLT /
// CPUID / RDTSC / LOCK-prefixed arithmetic / CMPXCHG8B/16B / CLC/STC/
// CMC/CLD/STD / PUSHF/POPF / LAHF/SAHF / LEAVE / ENTER.
bool HandleSystemLockInstruction(X64CpuState& s,
                                 const ZydisDecodedInstruction& inst,
                                 const ZydisDecodedOperand* operands);

// HLE call bridge (defined in x64_hle_bridge.cpp). The interpreter
// calls IsAerolibStubAddress(rip) on every CALL/JMP target; if it
// returns true, it calls DispatchAerolibStub instead of decoding
// ARM64 code from the stub arena.
bool IsAerolibStubAddress(u64 rip);
bool DispatchAerolibStub(u64 rip, X64CpuState* state);

} // namespace Core::Cpu
