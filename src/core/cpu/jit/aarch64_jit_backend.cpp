// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later
//
// Aarch64JitBackend implementation — x86-64 → ARM64 JIT compiler.
//
// This is the MVP JIT: it translates the most common x86-64 instructions
// (MOV, ADD, SUB, AND, OR, XOR, CMP, JMP, Jcc, CALL, RET, PUSH, POP, NOP,
// LEA, INC, DEC, NEG, NOT, SHL, SHR, SAR, MOVSX, MOVZX, CMOV, TEST) into
// ARM64 machine code and caches the result. Untranslated instructions
// fall back to the interpreter.
//
// The guest VA = host VA identity mapping means we can directly access
// guest memory from the translated ARM64 code — no address translation
// is needed in the hot path.
//
// Register mapping (fixed, no allocation):
//   x86-64 GPR[0..15] (RAX..R15) → ARM64 X0..X15
//   x86-64 RIP                → ARM64 X16
//   x86-64 RFLAGS             → ARM64 X17
//   X64CpuState*              → ARM64 X18 (callee-saved)
//   X19                       → block code pointer (callee-saved by trampoline)
//   X20                       → temporary for address computation
//   X21-X23                   → temporaries for flag conversion
//
// X64CpuState layout:
//   offset 0:    u64 gpr[16]    (128 bytes)
//   offset 128:  u64 rip
//   offset 136:  u64 rflags
//
// Entry/exit trampoline (emitted once at init):
//   Entry:  saves X19/X20/X29/X30, sets X18=state & X19=block_code,
//           loads X0-X15 from gpr[], X16=rip, X17=rflags, sets X30 to
//           the exit handler, BR X19.
//   Exit:   stores X0-X15 back to gpr[], X16→rip, X17→rflags, restores
//           X19/X20/X29/X30, MOV X0,X16 (return value), RET.

#include "core/cpu/jit/aarch64_jit_backend.h"

#include <cstring>
#include <sys/mman.h>
#include <unistd.h>

#include <unordered_map>
#include <vector>

#include <Zydis/Zydis.h>

#include "common/logging/log.h"
#include "core/cpu/jit/arm64_emitter.h"
#include "core/cpu/interpreter/x64_interpreter_backend.h"
#include "core/cpu/x64_cpu_state.h"

namespace Core::Cpu {

using namespace Jit;

// ─────────────────────────────────────────────────────────────────────────
//  Zydis register → ARM64 register mapping
// ─────────────────────────────────────────────────────────────────────────

// Returns true iff `reg` is one of the 16 64-bit GPRs (RAX..R15) in Zydis
// order. The JIT only translates full 64-bit register operands; sub-register
// accesses (EAX/AX/AL) fall back to the interpreter.
static inline bool IsGpr64(ZydisRegister reg) {
    return reg >= ZYDIS_REGISTER_RAX && reg <= ZYDIS_REGISTER_R15;
}

// Maps a 64-bit Zydis GPR to its corresponding ARM64 X register.
// Caller must ensure `reg` is a 64-bit GPR (use IsGpr64 first).
static inline Arm64Reg ZydisGprToArm64(ZydisRegister reg) {
    return Arm64Reg(reg - ZYDIS_REGISTER_RAX);
}

// ─────────────────────────────────────────────────────────────────────────
//  Trampoline + block translator
// ─────────────────────────────────────────────────────────────────────────

// The trampoline function signature: u64 RunBlock(X64CpuState* state, void* block_code).
// The trampoline loads the GPRs/RIP/RFLAGS from `state` into X0-X17, jumps to
// `block_code`, and on return stores everything back. The new RIP is returned
// in X0.
using RunBlockFn = u64 (*)(X64CpuState* state, void* block_code);

// Per-block link record. When translating a block we emit a placeholder B
// (branch immediate) for any direct JMP/CALL to a guest RIP whose block is
// already cached. After AllocBlock gives us the host address, we patch each
// placeholder with the real offset to the target's host code.
struct LinkPatch {
    size_t buffer_offset;  // Offset of the B instruction within the buffer
    u64    target_rip;     // Guest RIP of the target block
};

struct Aarch64JitBackend::Impl {
    BlockCache block_cache;
    ZydisDecoder decoder;

    // Shared JIT CPU state. The trampoline reads/writes this; the JIT's
    // Execute() also uses it as the canonical view of the guest state for
    // fallback handling.
    X64CpuState state{};

    // The trampoline: mmap'd executable page holding the entry+exit code.
    void* trampoline_code = nullptr;
    RunBlockFn run_block = nullptr;

    // Tracks blocks that exit at an unimplemented instruction. Maps the
    // block's guest RIP → the guest RIP of the unimplemented instruction.
    // Execute() checks this map after a block returns; if the returned
    // RIP matches, it calls the interpreter for the rest of the block.
    std::unordered_map<u64, u64> fallback_rips;

    Impl() {
        ZydisDecoderInit(&decoder, ZYDIS_MACHINE_MODE_LONG_64, ZYDIS_STACK_WIDTH_64);
        EmitTrampoline();
    }

    ~Impl() {
        if (trampoline_code) {
            munmap(trampoline_code, 4096);
            trampoline_code = nullptr;
        }
    }

    // ── Trampoline emission ──
    // Emits the entry+exit trampoline into a fresh mmap'd RWX page and
    // stores the function pointer in `run_block`. The trampoline layout:
    //
    //   Entry:
    //     SUB  SP, SP, #32               ; allocate callee-save spill area
    //     STP  X19, X20, [SP, #0]        ; save callee-saved X19/X20
    //     STP  X29, X30, [SP, #16]       ; save FP/LR
    //     MOV  X18, X0                    ; X18 = X64CpuState*
    //     MOV  X19, X1                    ; X19 = block code pointer
    //     LDP  X0, X1, [X18, #0]          ; load RAX/RCX
    //     LDP  X2, X3, [X18, #16]        ; load RDX/RBX
    //     LDP  X4, X5, [X18, #32]        ; load RSP/RBP
    //     LDP  X6, X7, [X18, #48]        ; load RSI/RDI
    //     LDP  X8, X9, [X18, #64]        ; load R8/R9
    //     LDP  X10,X11,[X18, #80]        ; load R10/R11
    //     LDP  X12,X13,[X18, #96]        ; load R12/R13
    //     LDP  X14,X15,[X18,#112]        ; load R14/R15
    //     LDR  X16,[X18, #128]           ; load RIP
    //     LDR  X17,[X18, #136]           ; load RFLAGS
    //     ADR  X30, exit_handler          ; LR = exit handler address
    //     BR   X19                        ; jump to translated block
    //
    //   Exit handler (block does RET → lands here):
    //     STP  X0, X1, [X18, #0]          ; store RAX/RCX
    //     STP  X2, X3, [X18, #16]        ; store RDX/RBX
    //     STP  X4, X5, [X18, #32]        ; store RSP/RBP
    //     STP  X6, X7, [X18, #48]        ; store RSI/RDI
    //     STP  X8, X9, [X18, #64]        ; store R8/R9
    //     STP  X10,X11,[X18, #80]        ; store R10/R11
    //     STP  X12,X13,[X18, #96]        ; store R12/R13
    //     STP  X14,X15,[X18,#112]        ; store R14/R15
    //     STR  X16,[X18, #128]           ; store RIP
    //     STR  X17,[X18, #136]           ; store RFLAGS
    //     LDP  X29, X30, [SP, #16]       ; restore FP/LR
    //     LDP  X19, X20, [SP, #0]        ; restore X19/X20
    //     ADD  SP, SP, #32               ; deallocate spill area
    //     MOV  X0, X16                    ; return value = new RIP
    //     RET
    void EmitTrampoline() {
        // Allocate a 4 KB RWX page for the trampoline.
        trampoline_code = mmap(nullptr, 4096,
                               PROT_READ | PROT_WRITE | PROT_EXEC,
                               MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (trampoline_code == MAP_FAILED) {
            trampoline_code = nullptr;
            LOG_CRITICAL(Core_Cpu, "JIT: failed to mmap trampoline page");
            return;
        }

        CodeBuffer buf(4096);

        // ── Entry ──
        buf.SubImm(SP, SP, 32);                   // SUB  SP, SP, #32
        buf.Stp64(X19, X20, SP, 0);               // STP  X19, X20, [SP, #0]
        buf.Stp64(X29, X30, SP, 16);              // STP  X29, X30, [SP, #16]
        buf.Mov(X18, X0);                          // MOV  X18, X0  (state pointer)
        buf.Mov(X19, X1);                          // MOV  X19, X1  (block code)

        // Load X0-X15 from gpr[] in four pairs each, 8 bytes apart.
        buf.Ldp64(X0,  X1,  X18, 0);               // LDP X0,  X1,  [X18]
        buf.Ldp64(X2,  X3,  X18, 16);              // LDP X2,  X3,  [X18, #16]
        buf.Ldp64(X4,  X5,  X18, 32);              // LDP X4,  X5,  [X18, #32]
        buf.Ldp64(X6,  X7,  X18, 48);              // LDP X6,  X7,  [X18, #48]
        buf.Ldp64(X8,  X9,  X18, 64);              // LDP X8,  X9,  [X18, #64]
        buf.Ldp64(X10, X11, X18, 80);              // LDP X10, X11, [X18, #80]
        buf.Ldp64(X12, X13, X18, 96);              // LDP X12, X13, [X18, #96]
        buf.Ldp64(X14, X15, X18, 112);             // LDP X14, X15, [X18, #112]

        // Load RIP (X16) and RFLAGS (X17).
        buf.Ldr64(X16, X18, 128);                  // LDR X16, [X18, #128]
        buf.Ldr64(X17, X18, 136);                  // LDR X17, [X18, #136]

        // ADR X30, exit_handler — emitted with placeholder offset 0; we
        // patch the offset once we know where the exit handler lands.
        const s32 adr_offset = buf.current_offset();
        buf.Adr(X30, 0);

        // BR X19 — jump to the translated block. The block's first RET will
        // return to the exit handler (since we set X30 = exit_handler).
        buf.Br(X19);

        // ── Exit handler ──
        const s32 exit_offset = buf.current_offset();

        // Store X0-X15 back to gpr[] in pairs.
        buf.Stp64(X0,  X1,  X18, 0);
        buf.Stp64(X2,  X3,  X18, 16);
        buf.Stp64(X4,  X5,  X18, 32);
        buf.Stp64(X6,  X7,  X18, 48);
        buf.Stp64(X8,  X9,  X18, 64);
        buf.Stp64(X10, X11, X18, 80);
        buf.Stp64(X12, X13, X18, 96);
        buf.Stp64(X14, X15, X18, 112);

        // Store RIP and RFLAGS.
        buf.Str64(X16, X18, 128);
        buf.Str64(X17, X18, 136);

        // Restore callee-saved registers.
        buf.Ldp64(X29, X30, SP, 16);
        buf.Ldp64(X19, X20, SP, 0);
        buf.AddImm(SP, SP, 32);

        // Return value = new RIP. Then RET to the caller (Execute()).
        buf.Mov(X0, X16);
        buf.Ret();

        // Patch the ADR instruction with the real offset to the exit handler.
        const s32 adr_target_offset = exit_offset - adr_offset;
        const u32 adr_insn =
            0x10000000u
            | (((u32)adr_target_offset & 0x3u) << 29)
            | ((((u32)adr_target_offset >> 2) & 0x7FFFFu) << 5)
            | (u32)X30;
        // The CodeBuffer wrote a placeholder ADR at adr_offset. Rewrite it
        // in the buffer (data() returns const; we cast away const for the
        // in-place patch — the buffer is local and not shared).
        u8* base = const_cast<u8*>(buf.data());
        base[adr_offset + 0] = (u8)(adr_insn & 0xFF);
        base[adr_offset + 1] = (u8)((adr_insn >> 8) & 0xFF);
        base[adr_offset + 2] = (u8)((adr_insn >> 16) & 0xFF);
        base[adr_offset + 3] = (u8)((adr_insn >> 24) & 0xFF);

        // Copy the trampoline into the executable page and flush the
        // instruction cache (required on ARM64 so the CPU sees the new code).
        std::memcpy(trampoline_code, buf.data(), buf.size());
        __builtin___clear_cache(static_cast<char*>(trampoline_code),
                                static_cast<char*>(trampoline_code) + buf.size());

        run_block = reinterpret_cast<RunBlockFn>(trampoline_code);
        LOG_INFO(Core_Cpu, "JIT: trampoline emitted ({} bytes) at {}",
                 buf.size(), trampoline_code);
    }

    // ── Block translation ──
    // Translates one x86-64 basic block starting at `guest_rip` into ARM64
    // code, copies it into the executable code region, applies block-link
    // patches, and registers the block in the cache. Returns the block's
    // host entry point, or nullptr on failure.
    void* TranslateBlock(u64 guest_rip) {
        CodeBuffer buffer(4096);
        std::vector<LinkPatch> link_patches;

        u64 current_rip = guest_rip;
        u32 guest_bytes_translated = 0;
        bool block_terminated = false;
        bool hit_unimplemented = false;
        u64 unimplemented_rip = 0;

        while (!block_terminated) {
            // Read up to 15 bytes of guest code at current_rip. Identity
            // mapping means we can reinterpret the guest VA as a host ptr.
            const u8* code = reinterpret_cast<const u8*>(current_rip);
            u8 code_buf[15];
            // Guard against unmapped guest memory at end-of-region: copy
            // byte-by-byte and stop early on fault. For the MVP we just
            // memcpy; a real implementation would use the signal handler.
            std::memcpy(code_buf, code, 15);

            ZydisDecodedInstruction inst{};
            ZydisDecodedOperand operands[ZYDIS_MAX_OPERAND_COUNT];

            const ZyanStatus status = ZydisDecoderDecodeFull(
                &decoder, code_buf, 15, &inst, operands);
            if (!ZYAN_SUCCESS(status)) {
                // Can't decode — emit a fallback (set X16=current_rip, RET)
                // so Execute() can call the interpreter for this point.
                EmitSetX16Imm(buffer, current_rip);
                buf_Ret(buffer);
                hit_unimplemented = true;
                unimplemented_rip = current_rip;
                block_terminated = true;
                break;
            }

            const u64 next_rip = current_rip + inst.length;

            // Try to translate this instruction.
            TranslateResult tr =
                TranslateInstruction(buffer, current_rip, guest_rip, next_rip,
                                     inst, operands, link_patches);

            switch (tr) {
            case TranslateResult::kTranslated:
                current_rip = next_rip;
                guest_bytes_translated += (u32)inst.length;
                break;
            case TranslateResult::kTranslatedAndTerminated:
                guest_bytes_translated += (u32)inst.length;
                block_terminated = true;
                break;
            case TranslateResult::kUnimplemented:
                // Emit a fallback (set X16 = current_rip; RET) so the
                // block exits here and Execute() can call the interpreter
                // for this single instruction.
                EmitSetX16Imm(buffer, current_rip);
                buf_Ret(buffer);
                hit_unimplemented = true;
                unimplemented_rip = current_rip;
                block_terminated = true;
                break;
            }

            // If we didn't terminate explicitly, check if the instruction
            // is a known block terminator (JMP, RET, Jcc, CALL, HLT, ...).
            if (!block_terminated && IsBlockTerminator(inst.mnemonic)) {
                // The translator should have emitted a terminator; if it
                // didn't, force one here to keep the block bounded.
                EmitSetX16Imm(buffer, next_rip);
                buf_Ret(buffer);
                block_terminated = true;
            }
        }

        if (buffer.size() == 0) {
            return nullptr;
        }

        // Allocate space in the code region and copy the emitted code.
        void* host_code = block_cache.AllocBlock(guest_rip, buffer.size());
        if (!host_code) {
            return nullptr;
        }
        std::memcpy(host_code, buffer.data(), buffer.size());

        // Apply block-link patches: rewrite each placeholder B instruction
        // with a real B to the target block's host entry point.
        for (const LinkPatch& p : link_patches) {
            const BlockEntry* target = block_cache.Lookup(p.target_rip);
            if (target && target->valid && target->host_code) {
                u8* branch_addr = static_cast<u8*>(host_code) + p.buffer_offset;
                const u64 branch_host = reinterpret_cast<u64>(branch_addr);
                const u64 target_host = reinterpret_cast<u64>(target->host_code);
                const s32 offset = static_cast<s32>(target_host - branch_host);
                const u32 b_insn = 0x14000000u | ((static_cast<u32>(offset / 4)) & 0x03FFFFFFu);
                std::memcpy(branch_addr, &b_insn, sizeof(b_insn));
            }
            // If the target isn't cached yet, the placeholder B remains —
            // Execute() will look up the target on the next call and
            // re-enter the JIT. (No back-patching for the MVP.)
        }

        // Flush the instruction cache.
        __builtin___clear_cache(static_cast<char*>(host_code),
                                static_cast<char*>(host_code) + buffer.size());

        // Register the block.
        block_cache.FinalizeBlock(guest_rip, host_code, buffer.size(),
                                  guest_bytes_translated);

        // Record the fallback RIP for this block (if any).
        if (hit_unimplemented) {
            fallback_rips[guest_rip] = unimplemented_rip;
        }

        return host_code;
    }

    // Result of TranslateInstruction.
    enum class TranslateResult {
        kTranslated,            // Translated; continue to next instruction
        kTranslatedAndTerminated, // Translated; block ends here
        kUnimplemented,         // Cannot translate; emit fallback
    };

    // ── Single-instruction translator ──
    // Dispatches on the Zydis mnemonic to a per-instruction translator.
    TranslateResult TranslateInstruction(CodeBuffer& buf, u64 rip, u64 guest_rip,
                                          u64 next_rip,
                                          const ZydisDecodedInstruction& inst,
                                          const ZydisDecodedOperand* operands,
                                          std::vector<LinkPatch>& link_patches) {
        switch (inst.mnemonic) {
        case ZYDIS_MNEMONIC_NOP:
            buf.Nop();
            return TranslateResult::kTranslated;

        case ZYDIS_MNEMONIC_MOV:
            return TranslateMov(buf, rip, next_rip, inst, operands);

        case ZYDIS_MNEMONIC_ADD:
            return TranslateAdd(buf, inst, operands);

        case ZYDIS_MNEMONIC_SUB:
            return TranslateSub(buf, inst, operands);

        case ZYDIS_MNEMONIC_CMP:
            return TranslateCmp(buf, inst, operands);

        case ZYDIS_MNEMONIC_TEST:
            return TranslateTest(buf, inst, operands);

        case ZYDIS_MNEMONIC_AND:
            return TranslateAnd(buf, inst, operands);

        case ZYDIS_MNEMONIC_OR:
            return TranslateOr(buf, inst, operands);

        case ZYDIS_MNEMONIC_XOR:
            return TranslateXor(buf, inst, operands);

        case ZYDIS_MNEMONIC_LEA:
            return TranslateLea(buf, rip, next_rip, inst, operands);

        case ZYDIS_MNEMONIC_JMP:
            return TranslateJmp(buf, rip, next_rip, operands, link_patches);

        case ZYDIS_MNEMONIC_JO:
        case ZYDIS_MNEMONIC_JNO:
        case ZYDIS_MNEMONIC_JB:
        case ZYDIS_MNEMONIC_JNB:
        case ZYDIS_MNEMONIC_JZ:
        case ZYDIS_MNEMONIC_JNZ:
        case ZYDIS_MNEMONIC_JBE:
        case ZYDIS_MNEMONIC_JNBE:
        case ZYDIS_MNEMONIC_JS:
        case ZYDIS_MNEMONIC_JNS:
        case ZYDIS_MNEMONIC_JP:
        case ZYDIS_MNEMONIC_JNP:
        case ZYDIS_MNEMONIC_JL:
        case ZYDIS_MNEMONIC_JNL:
        case ZYDIS_MNEMONIC_JLE:
        case ZYDIS_MNEMONIC_JNLE:
            return TranslateJcc(buf, inst.mnemonic, rip, next_rip, operands);

        case ZYDIS_MNEMONIC_CALL:
            return TranslateCall(buf, rip, next_rip, operands, link_patches);

        case ZYDIS_MNEMONIC_RET:
            return TranslateRet(buf);

        case ZYDIS_MNEMONIC_PUSH:
            return TranslatePush(buf, operands);

        case ZYDIS_MNEMONIC_POP:
            return TranslatePop(buf, operands);

        case ZYDIS_MNEMONIC_INC:
            return TranslateInc(buf, inst, operands);

        case ZYDIS_MNEMONIC_DEC:
            return TranslateDec(buf, inst, operands);

        case ZYDIS_MNEMONIC_NEG:
            return TranslateNeg(buf, inst, operands);

        case ZYDIS_MNEMONIC_NOT:
            return TranslateNot(buf, inst, operands);

        case ZYDIS_MNEMONIC_MOVSX:
        case ZYDIS_MNEMONIC_MOVSXD:
            return TranslateMovsx(buf, inst, operands);

        case ZYDIS_MNEMONIC_MOVZX:
            return TranslateMovzx(buf, inst, operands);

        case ZYDIS_MNEMONIC_CMOVNB:
        case ZYDIS_MNEMONIC_CMOVNBE:
        case ZYDIS_MNEMONIC_CMOVZ:
        case ZYDIS_MNEMONIC_CMOVNZ:
        case ZYDIS_MNEMONIC_CMOVB:
        case ZYDIS_MNEMONIC_CMOVBE:
        case ZYDIS_MNEMONIC_CMOVL:
        case ZYDIS_MNEMONIC_CMOVLE:
        case ZYDIS_MNEMONIC_CMOVNL:
        case ZYDIS_MNEMONIC_CMOVNLE:
        case ZYDIS_MNEMONIC_CMOVS:
        case ZYDIS_MNEMONIC_CMOVNS:
        case ZYDIS_MNEMONIC_CMOVO:
        case ZYDIS_MNEMONIC_CMOVNO:
        case ZYDIS_MNEMONIC_CMOVP:
        case ZYDIS_MNEMONIC_CMOVNP:
            return TranslateCmov(buf, inst.mnemonic, inst, operands);

        case ZYDIS_MNEMONIC_SHL:
        case ZYDIS_MNEMONIC_SHR:
        case ZYDIS_MNEMONIC_SAR:
            return TranslateShift(buf, inst, operands);

        default:
            return TranslateResult::kUnimplemented;
        }
    }

    // ── Block terminator detection ──
    bool IsBlockTerminator(ZydisMnemonic mn) {
        switch (mn) {
        case ZYDIS_MNEMONIC_JMP:
        case ZYDIS_MNEMONIC_RET:
        case ZYDIS_MNEMONIC_CALL:
        case ZYDIS_MNEMONIC_JO: case ZYDIS_MNEMONIC_JNO:
        case ZYDIS_MNEMONIC_JB: case ZYDIS_MNEMONIC_JNB:
        case ZYDIS_MNEMONIC_JZ: case ZYDIS_MNEMONIC_JNZ:
        case ZYDIS_MNEMONIC_JBE: case ZYDIS_MNEMONIC_JNBE:
        case ZYDIS_MNEMONIC_JS: case ZYDIS_MNEMONIC_JNS:
        case ZYDIS_MNEMONIC_JP: case ZYDIS_MNEMONIC_JNP:
        case ZYDIS_MNEMONIC_JL: case ZYDIS_MNEMONIC_JNL:
        case ZYDIS_MNEMONIC_JLE: case ZYDIS_MNEMONIC_JNLE:
        case ZYDIS_MNEMONIC_JCXZ: case ZYDIS_MNEMONIC_JECXZ:
        case ZYDIS_MNEMONIC_JRCXZ:
        case ZYDIS_MNEMONIC_HLT:
        case ZYDIS_MNEMONIC_UD2:
        case ZYDIS_MNEMONIC_INT3:
        case ZYDIS_MNEMONIC_INT1:
        case ZYDIS_MNEMONIC_INT:
            return true;
        default:
            return false;
        }
    }

    // =====================================================================
    //  Instruction translators
    // =====================================================================

    // ── MOV ──
    TranslateResult TranslateMov(CodeBuffer& buf, u64 rip, u64 next_rip,
                                    const ZydisDecodedInstruction& inst,
                                    const ZydisDecodedOperand* operands) {
        // MOV reg, reg
        if (operands[0].type == ZYDIS_OPERAND_TYPE_REGISTER &&
            operands[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
            if (!IsGpr64(operands[0].reg.value) ||
                !IsGpr64(operands[1].reg.value)) {
                return TranslateResult::kUnimplemented;
            }
            const Arm64Reg dst = ZydisGprToArm64(operands[0].reg.value);
            const Arm64Reg src = ZydisGprToArm64(operands[1].reg.value);
            buf.Mov(dst, src);
            return TranslateResult::kTranslated;
        }

        // MOV reg, imm
        if (operands[0].type == ZYDIS_OPERAND_TYPE_REGISTER &&
            operands[1].type == ZYDIS_OPERAND_TYPE_IMMEDIATE) {
            if (!IsGpr64(operands[0].reg.value)) {
                return TranslateResult::kUnimplemented;
            }
            const Arm64Reg dst = ZydisGprToArm64(operands[0].reg.value);
            buf.MovImm64(dst, operands[1].imm.value.u);
            return TranslateResult::kTranslated;
        }

        // MOV reg, [mem]
        if (operands[0].type == ZYDIS_OPERAND_TYPE_REGISTER &&
            operands[1].type == ZYDIS_OPERAND_TYPE_MEMORY) {
            if (!IsGpr64(operands[0].reg.value) ||
                operands[0].size != 64) {
                return TranslateResult::kUnimplemented;
            }
            const Arm64Reg dst = ZydisGprToArm64(operands[0].reg.value);
            return EmitLoadMem(buf, dst, operands[1], rip, next_rip);
        }

        // MOV [mem], reg
        if (operands[0].type == ZYDIS_OPERAND_TYPE_MEMORY &&
            operands[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
            if (!IsGpr64(operands[1].reg.value) ||
                operands[1].size != 64) {
                return TranslateResult::kUnimplemented;
            }
            const Arm64Reg src = ZydisGprToArm64(operands[1].reg.value);
            return EmitStoreMem(buf, src, operands[0], rip, next_rip);
        }

        // MOV [mem], imm — MVP: fall back to interpreter.
        return TranslateResult::kUnimplemented;
    }

    // ── ADD ──
    TranslateResult TranslateAdd(CodeBuffer& buf,
                                   const ZydisDecodedInstruction& inst,
                                   const ZydisDecodedOperand* operands) {
        // ADD reg, reg
        if (operands[0].type == ZYDIS_OPERAND_TYPE_REGISTER &&
            operands[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
            if (!IsGpr64(operands[0].reg.value) ||
                !IsGpr64(operands[1].reg.value)) {
                return TranslateResult::kUnimplemented;
            }
            const Arm64Reg dst = ZydisGprToArm64(operands[0].reg.value);
            const Arm64Reg src = ZydisGprToArm64(operands[1].reg.value);
            buf.Add(dst, dst, src);
            // Note: ADD without flag-setting. The MVP doesn't update RFLAGS
            // here because Zydis marks the operand as "writes flags" only for
            // ADD-form instructions that set the flags. A full implementation
            // would emit ADDS + flag conversion when the mnemonic has the
            // flag-setting attribute.
            return TranslateResult::kTranslated;
        }

        // ADD reg, imm
        if (operands[0].type == ZYDIS_OPERAND_TYPE_REGISTER &&
            operands[1].type == ZYDIS_OPERAND_TYPE_IMMEDIATE) {
            if (!IsGpr64(operands[0].reg.value)) {
                return TranslateResult::kUnimplemented;
            }
            const Arm64Reg dst = ZydisGprToArm64(operands[0].reg.value);
            const u64 imm = operands[1].imm.value.u;
            if (imm <= 0xFFF) {
                buf.AddImm(dst, dst, (u32)imm);
                return TranslateResult::kTranslated;
            }
            // For larger immediates, materialise the constant in X21 and ADD.
            buf.MovImm64(X21, imm);
            buf.Add(dst, dst, X21);
            return TranslateResult::kTranslated;
        }

        return TranslateResult::kUnimplemented;
    }

    // ── SUB ──
    TranslateResult TranslateSub(CodeBuffer& buf,
                                   const ZydisDecodedInstruction& inst,
                                   const ZydisDecodedOperand* operands) {
        if (operands[0].type == ZYDIS_OPERAND_TYPE_REGISTER &&
            operands[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
            if (!IsGpr64(operands[0].reg.value) ||
                !IsGpr64(operands[1].reg.value)) {
                return TranslateResult::kUnimplemented;
            }
            const Arm64Reg dst = ZydisGprToArm64(operands[0].reg.value);
            const Arm64Reg src = ZydisGprToArm64(operands[1].reg.value);
            buf.Sub(dst, dst, src);
            return TranslateResult::kTranslated;
        }

        if (operands[0].type == ZYDIS_OPERAND_TYPE_REGISTER &&
            operands[1].type == ZYDIS_OPERAND_TYPE_IMMEDIATE) {
            if (!IsGpr64(operands[0].reg.value)) {
                return TranslateResult::kUnimplemented;
            }
            const Arm64Reg dst = ZydisGprToArm64(operands[0].reg.value);
            const u64 imm = operands[1].imm.value.u;
            if (imm <= 0xFFF) {
                buf.SubImm(dst, dst, (u32)imm);
                return TranslateResult::kTranslated;
            }
            buf.MovImm64(X21, imm);
            buf.Sub(dst, dst, X21);
            return TranslateResult::kTranslated;
        }

        return TranslateResult::kUnimplemented;
    }

    // ── CMP (SUBS XZR, Xn, Xm — sets NZCV, then convert to RFLAGS) ──
    TranslateResult TranslateCmp(CodeBuffer& buf,
                                    const ZydisDecodedInstruction& inst,
                                    const ZydisDecodedOperand* operands) {
        if (operands[0].type == ZYDIS_OPERAND_TYPE_REGISTER &&
            operands[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
            if (!IsGpr64(operands[0].reg.value) ||
                !IsGpr64(operands[1].reg.value)) {
                return TranslateResult::kUnimplemented;
            }
            const Arm64Reg lhs = ZydisGprToArm64(operands[0].reg.value);
            const Arm64Reg rhs = ZydisGprToArm64(operands[1].reg.value);
            buf.Cmp(lhs, rhs);  // SUBS XZR, lhs, rhs → sets NZCV
            EmitConvertFlags(buf, /*is_sub=*/true);
            return TranslateResult::kTranslated;
        }

        if (operands[0].type == ZYDIS_OPERAND_TYPE_REGISTER &&
            operands[1].type == ZYDIS_OPERAND_TYPE_IMMEDIATE) {
            if (!IsGpr64(operands[0].reg.value)) {
                return TranslateResult::kUnimplemented;
            }
            const Arm64Reg lhs = ZydisGprToArm64(operands[0].reg.value);
            const u64 imm = operands[1].imm.value.u;
            // The emitter has no SUBS-immediate form, so materialise the
            // immediate in X21 and use the register-form SUBS.
            buf.MovImm64(X21, imm);
            buf.Subs(Arm64Reg::ZR, lhs, X21);  // SUBS XZR, lhs, X21 → sets NZCV
            EmitConvertFlags(buf, /*is_sub=*/true);
            return TranslateResult::kTranslated;
        }

        return TranslateResult::kUnimplemented;
    }

    // ── TEST (ANDS XZR, Xn, Xm — sets N/Z; C=0, V=0) ──
    TranslateResult TranslateTest(CodeBuffer& buf,
                                     const ZydisDecodedInstruction& inst,
                                     const ZydisDecodedOperand* operands) {
        if (operands[0].type == ZYDIS_OPERAND_TYPE_REGISTER &&
            operands[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
            if (!IsGpr64(operands[0].reg.value) ||
                !IsGpr64(operands[1].reg.value)) {
                return TranslateResult::kUnimplemented;
            }
            const Arm64Reg lhs = ZydisGprToArm64(operands[0].reg.value);
            const Arm64Reg rhs = ZydisGprToArm64(operands[1].reg.value);
            // ANDS XZR, lhs, rhs → sets NZCV (N, Z from result; C=0, V=0)
            buf.Ands(Arm64Reg::ZR, lhs, rhs);
            EmitConvertFlags(buf, /*is_sub=*/false);
            return TranslateResult::kTranslated;
        }
        return TranslateResult::kUnimplemented;
    }

    // ── AND ──
    TranslateResult TranslateAnd(CodeBuffer& buf,
                                    const ZydisDecodedInstruction& inst,
                                    const ZydisDecodedOperand* operands) {
        if (operands[0].type == ZYDIS_OPERAND_TYPE_REGISTER &&
            operands[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
            if (!IsGpr64(operands[0].reg.value) ||
                !IsGpr64(operands[1].reg.value)) {
                return TranslateResult::kUnimplemented;
            }
            const Arm64Reg dst = ZydisGprToArm64(operands[0].reg.value);
            const Arm64Reg src = ZydisGprToArm64(operands[1].reg.value);
            buf.And(dst, dst, src);
            return TranslateResult::kTranslated;
        }
        return TranslateResult::kUnimplemented;
    }

    // ── OR ──
    TranslateResult TranslateOr(CodeBuffer& buf,
                                   const ZydisDecodedInstruction& inst,
                                   const ZydisDecodedOperand* operands) {
        if (operands[0].type == ZYDIS_OPERAND_TYPE_REGISTER &&
            operands[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
            if (!IsGpr64(operands[0].reg.value) ||
                !IsGpr64(operands[1].reg.value)) {
                return TranslateResult::kUnimplemented;
            }
            const Arm64Reg dst = ZydisGprToArm64(operands[0].reg.value);
            const Arm64Reg src = ZydisGprToArm64(operands[1].reg.value);
            buf.Orr(dst, dst, src);
            return TranslateResult::kTranslated;
        }
        return TranslateResult::kUnimplemented;
    }

    // ── XOR ──
    TranslateResult TranslateXor(CodeBuffer& buf,
                                    const ZydisDecodedInstruction& inst,
                                    const ZydisDecodedOperand* operands) {
        if (operands[0].type == ZYDIS_OPERAND_TYPE_REGISTER &&
            operands[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
            if (!IsGpr64(operands[0].reg.value) ||
                !IsGpr64(operands[1].reg.value)) {
                return TranslateResult::kUnimplemented;
            }
            const Arm64Reg dst = ZydisGprToArm64(operands[0].reg.value);
            const Arm64Reg src = ZydisGprToArm64(operands[1].reg.value);
            if (dst == src) {
                // XOR reg, reg → zero the register (and set ZF=1, SF=0, ...).
                buf.MovImm64(dst, 0);
                // XOR sets all logical flags: ZF=1, others=0.
                buf.MovImm64(X17, RflagsBits::ZF);
                return TranslateResult::kTranslated;
            }
            buf.Eor(dst, dst, src);
            return TranslateResult::kTranslated;
        }
        return TranslateResult::kUnimplemented;
    }

    // ── LEA reg, [mem] ──
    TranslateResult TranslateLea(CodeBuffer& buf, u64 rip, u64 next_rip,
                                    const ZydisDecodedInstruction& inst,
                                    const ZydisDecodedOperand* operands) {
        if (operands[0].type != ZYDIS_OPERAND_TYPE_REGISTER ||
            operands[1].type != ZYDIS_OPERAND_TYPE_MEMORY) {
            return TranslateResult::kUnimplemented;
        }
        if (!IsGpr64(operands[0].reg.value)) {
            return TranslateResult::kUnimplemented;
        }
        const Arm64Reg dst = ZydisGprToArm64(operands[0].reg.value);
        if (!EmitLeaMem(buf, dst, operands[1], rip, next_rip)) {
            return TranslateResult::kUnimplemented;
        }
        return TranslateResult::kTranslated;
    }

    // ── JMP ──
    TranslateResult TranslateJmp(CodeBuffer& buf, u64 rip, u64 next_rip,
                                    const ZydisDecodedOperand* operands,
                                    std::vector<LinkPatch>& link_patches) {
        // JMP imm — direct jump.
        if (operands[0].type == ZYDIS_OPERAND_TYPE_IMMEDIATE) {
            const u64 target = next_rip + operands[0].imm.value.s;
            // Try to link directly to a cached target block. We emit a
            // placeholder B and record a LinkPatch; after AllocBlock we
            // rewrite it with the real offset. If the target isn't cached
            // yet, we leave a fallback MOV X16,target; RET and skip the
            // patch (the placeholder B will never be patched, but the
            // MOV+RET will already have exited the block).
            const BlockEntry* target_block = block_cache.Lookup(target);
            if (target_block && target_block->valid) {
                // Target cached but we don't yet know our own host address
                // — emit a placeholder B and patch later. We must still
                // fall through to a MOV+RET in case linking fails, so we
                // emit B (placeholder) followed by MOV X16,target; RET.
                // The placeholder B will be patched; if patching fails
                // the B stays at offset 0 → infinite loop, so we make
                // sure to ALWAYS succeed in patching by checking here.
                const s32 b_offset = buf.current_offset();
                buf.B(0);  // placeholder
                link_patches.push_back({(size_t)b_offset, target});
                return TranslateResult::kTranslatedAndTerminated;
            }
            // Target not cached — emit MOV X16,target; RET (exit to host).
            EmitSetX16Imm(buf, target);
            buf_Ret(buf);
            return TranslateResult::kTranslatedAndTerminated;
        }

        // JMP [reg] — indirect jump.
        if (operands[0].type == ZYDIS_OPERAND_TYPE_MEMORY) {
            // Compute the effective address into X16, then load RIP from
            // that address. We need a separate temp because LDR can't
            // load directly into X16 if the addressing uses X16.
            // (RIP-relative addressing would touch X16 — handle via X20.)
            // Use X20 as the address temporary, then LDR into X16.
            if (!EmitLeaMem(buf, X20, operands[0], rip, next_rip)) {
                return TranslateResult::kUnimplemented;
            }
            buf.Ldr64(X16, X20, 0);
            buf_Ret(buf);
            return TranslateResult::kTranslatedAndTerminated;
        }

        return TranslateResult::kUnimplemented;
    }

    // ── Jcc ──
    // For each condition, extract the relevant flag bit(s) from X17 (RFLAGS)
    // using UBFX+CBNZ or UBFX+CBZ, then branch to either the target RIP
    // (taken) or the next RIP (fall-through). Both paths end with
    // MOV X16, rip; RET — the block exits either way, and Execute() picks
    // up at the new RIP.
    //
    // Emitted layout:
    //   <condition test>     ; extracts flag bit(s) into X21
    //   CBZ/CBNZ X21, taken   ; branch if condition true (forward, patched)
    //   MOV X16, next_rip     ; fall-through: X16 = next RIP
    //   RET
    // taken:
    //   MOV X16, target      ; taken: X16 = target RIP
    //   RET
    TranslateResult TranslateJcc(CodeBuffer& buf, ZydisMnemonic mn,
                                    u64 rip, u64 next_rip,
                                    const ZydisDecodedOperand* operands) {
        if (operands[0].type != ZYDIS_OPERAND_TYPE_IMMEDIATE) {
            return TranslateResult::kUnimplemented;
        }
        const u64 target = next_rip + operands[0].imm.value.s;

        // Emit the condition test (extracts flag bits into X21).
        // Returns true for "flag == 0" conditions (use CBZ), false for
        // "flag == 1" conditions (use CBNZ).
        const bool inverted = EmitJccTest(buf, mn);

        // Emit placeholder CBZ/CBNZ — patched with the real offset once
        // we know where the taken path lands.
        const s32 branch_offset = buf.current_offset();
        if (inverted) {
            buf.Cbz(X21, 0);
        } else {
            buf.Cbnz(X21, 0);
        }

        // Fall-through path: X16 = next_rip, then RET (returns to trampoline
        // exit handler). The trampoline stores X16 → X64CpuState::rip.
        EmitSetX16Imm(buf, next_rip);
        buf_Ret(buf);

        // Taken path: X16 = target, then RET.
        const s32 taken_offset = buf.current_offset();
        EmitSetX16Imm(buf, target);
        buf_Ret(buf);

        // Patch the CBZ/CBNZ with the offset to the taken path.
        const s32 cb_offset = taken_offset - branch_offset;
        const u32 cb_base = inverted ? 0xB4000000u : 0xB5000000u;
        const u32 cb_insn = cb_base
                         | (((u32)(cb_offset / 4) & 0x7FFFFu) << 5)
                         | (u32)X21;
        u8* base = const_cast<u8*>(buf.data());
        base[branch_offset + 0] = (u8)(cb_insn & 0xFF);
        base[branch_offset + 1] = (u8)((cb_insn >> 8) & 0xFF);
        base[branch_offset + 2] = (u8)((cb_insn >> 16) & 0xFF);
        base[branch_offset + 3] = (u8)((cb_insn >> 24) & 0xFF);

        return TranslateResult::kTranslatedAndTerminated;
    }

    // Emit the condition test for a Jcc. Returns true if the condition
    // is "flag == 0" (i.e., the branch should be CBZ); false if "flag == 1"
    // (i.e., the branch should be CBNZ). After this, X21 holds 1 if the
    // condition is true, 0 otherwise.
    bool EmitJccTest(CodeBuffer& buf, ZydisMnemonic mn) {
        switch (mn) {
        // JO — OF = 1
        case ZYDIS_MNEMONIC_JO:
            buf.Ubfx(X21, X17, 11, 1);
            return false;
        // JNO — OF = 0
        case ZYDIS_MNEMONIC_JNO:
            buf.Ubfx(X21, X17, 11, 1);
            return true;
        // JB / JC — CF = 1
        case ZYDIS_MNEMONIC_JB:
            buf.Ubfx(X21, X17, 0, 1);
            return false;
        // JNB / JNC — CF = 0
        case ZYDIS_MNEMONIC_JNB:
            buf.Ubfx(X21, X17, 0, 1);
            return true;
        // JZ / JE — ZF = 1
        case ZYDIS_MNEMONIC_JZ:
            buf.Ubfx(X21, X17, 6, 1);
            return false;
        // JNZ / JNE — ZF = 0
        case ZYDIS_MNEMONIC_JNZ:
            buf.Ubfx(X21, X17, 6, 1);
            return true;
        // JBE — CF || ZF
        case ZYDIS_MNEMONIC_JBE: {
            buf.Ubfx(X21, X17, 0, 1);  // CF
            buf.Ubfx(X22, X17, 6, 1); // ZF
            buf.Orr(X21, X21, X22);
            return false;
        }
        // JNBE / JA — !CF && !ZF  → branch if (CF || ZF) == 0
        case ZYDIS_MNEMONIC_JNBE: {
            buf.Ubfx(X21, X17, 0, 1);  // CF
            buf.Ubfx(X22, X17, 6, 1); // ZF
            buf.Orr(X21, X21, X22);
            return true;
        }
        // JS — SF = 1
        case ZYDIS_MNEMONIC_JS:
            buf.Ubfx(X21, X17, 7, 1);
            return false;
        // JNS — SF = 0
        case ZYDIS_MNEMONIC_JNS:
            buf.Ubfx(X21, X17, 7, 1);
            return true;
        // JP — PF = 1
        case ZYDIS_MNEMONIC_JP:
            buf.Ubfx(X21, X17, 2, 1);
            return false;
        // JNP — PF = 0
        case ZYDIS_MNEMONIC_JNP:
            buf.Ubfx(X21, X17, 2, 1);
            return true;
        // JL — SF != OF
        case ZYDIS_MNEMONIC_JL: {
            buf.Ubfx(X21, X17, 7, 1);   // SF
            buf.Ubfx(X22, X17, 11, 1); // OF
            buf.Eor(X21, X21, X22);
            return false;
        }
        // JNL / JGE — SF == OF  → branch if (SF XOR OF) == 0
        case ZYDIS_MNEMONIC_JNL: {
            buf.Ubfx(X21, X17, 7, 1);   // SF
            buf.Ubfx(X22, X17, 11, 1); // OF
            buf.Eor(X21, X21, X22);
            return true;
        }
        // JLE — ZF || (SF != OF)
        case ZYDIS_MNEMONIC_JLE: {
            buf.Ubfx(X21, X17, 6, 1);   // ZF
            buf.Ubfx(X22, X17, 7, 1);   // SF
            buf.Ubfx(X23, X17, 11, 1);  // OF
            buf.Eor(X22, X22, X23);     // SF XOR OF
            buf.Orr(X21, X21, X22);     // ZF || (SF XOR OF)
            return false;
        }
        // JNLE / JG — !ZF && (SF == OF)  → branch if (ZF || (SF XOR OF)) == 0
        case ZYDIS_MNEMONIC_JNLE: {
            buf.Ubfx(X21, X17, 6, 1);   // ZF
            buf.Ubfx(X22, X17, 7, 1);   // SF
            buf.Ubfx(X23, X17, 11, 1);  // OF
            buf.Eor(X22, X22, X23);     // SF XOR OF
            buf.Orr(X21, X21, X22);     // ZF || (SF XOR OF)
            return true;
        }
        default:
            // Unknown Jcc — branch never (X21 = 0).
            buf.MovImm64(X21, 0);
            return true;
        }
    }

    // Maps a Jcc mnemonic to the ARM64 condition code that would
    // implement it directly (used by the placeholder approach above).
    static u32 EmitJccCondition(ZydisMnemonic mn) {
        // This is no longer used for the real Jcc path; we use UBFX+CBNZ
        // instead so we can read from X17 (which holds the x86 RFLAGS).
        // Returned for completeness; not called by the real path.
        switch (mn) {
        case ZYDIS_MNEMONIC_JO:   return (u32)Cond::VS;
        case ZYDIS_MNEMONIC_JNO:  return (u32)Cond::VC;
        case ZYDIS_MNEMONIC_JB:   return (u32)Cond::CS;
        case ZYDIS_MNEMONIC_JNB:  return (u32)Cond::CC;
        case ZYDIS_MNEMONIC_JZ:   return (u32)Cond::EQ;
        case ZYDIS_MNEMONIC_JNZ:  return (u32)Cond::NE;
        case ZYDIS_MNEMONIC_JBE:  return (u32)Cond::LS;
        case ZYDIS_MNEMONIC_JNBE: return (u32)Cond::HI;
        case ZYDIS_MNEMONIC_JS:   return (u32)Cond::MI;
        case ZYDIS_MNEMONIC_JNS:  return (u32)Cond::PL;
        case ZYDIS_MNEMONIC_JP:   return (u32)Cond::AL;  // PF not in ARM64 NZCV
        case ZYDIS_MNEMONIC_JNP:  return (u32)Cond::NV;
        case ZYDIS_MNEMONIC_JL:   return (u32)Cond::LT;
        case ZYDIS_MNEMONIC_JNL:  return (u32)Cond::GE;
        case ZYDIS_MNEMONIC_JLE:  return (u32)Cond::LE;
        case ZYDIS_MNEMONIC_JNLE: return (u32)Cond::GT;
        default:                  return (u32)Cond::AL;
        }
    }

    // ── CALL ──
    // Push return address (next_rip) onto the stack, then set X16 = target
    // and exit the block. Execute() will pick up at the target on the next
    // call. The return address is popped by RET.
    TranslateResult TranslateCall(CodeBuffer& buf, u64 rip, u64 next_rip,
                                     const ZydisDecodedOperand* operands,
                                     std::vector<LinkPatch>& link_patches) {
        if (operands[0].type != ZYDIS_OPERAND_TYPE_IMMEDIATE) {
            // Indirect CALL — fall back for the MVP.
            return TranslateResult::kUnimplemented;
        }
        const u64 target = next_rip + operands[0].imm.value.s;

        // The block's CALL pushes the return address (next_rip) and jumps
        // to `target`. We materialise next_rip into X16 directly because
        // X16 may not track the current RIP continuously — at this point
        // X16 still holds the block-start RIP (set by the trampoline), not
        // the CALL's RIP. Using the absolute next_rip ensures the right
        // value is pushed regardless of where the CALL sits in the block.
        EmitSetX16Imm(buf, next_rip);              // X16 = next_rip (return addr)
        buf.Str64PreIndex(X16, X4, 8);            // STR X16, [X4, #-8]! (push)
        EmitSetX16Imm(buf, target);                // X16 = target (new RIP)
        // Try block linking (same as JMP).
        const BlockEntry* target_block = block_cache.Lookup(target);
        if (target_block && target_block->valid) {
            const s32 b_offset = buf.current_offset();
            buf.B(0);  // placeholder — patched in TranslateBlock after AllocBlock.
            link_patches.push_back({(size_t)b_offset, target});
            // The B replaces the RET below in the patched binary; we still
            // emit a RET as the unpatched fallback (so unpatched runs return
            // to Execute() instead of falling off the end).
            buf_Ret(buf);
            return TranslateResult::kTranslatedAndTerminated;
        }
        buf_Ret(buf);
        return TranslateResult::kTranslatedAndTerminated;
    }

    // ── RET ──
    // Pop the return address from the stack into X16, then exit the block.
    TranslateResult TranslateRet(CodeBuffer& buf) {
        // LDR X16, [X4], #8 — pop return address (X4 = RSP, post-increment).
        buf.Ldr64PostIndex(X16, X4, 8);
        buf_Ret(buf);
        return TranslateResult::kTranslatedAndTerminated;
    }

    // ── PUSH ──
    // PUSH reg: STR Xreg, [X4, #-8]! (pre-decrement RSP, then store).
    TranslateResult TranslatePush(CodeBuffer& buf,
                                     const ZydisDecodedOperand* operands) {
        if (operands[0].type != ZYDIS_OPERAND_TYPE_REGISTER ||
            !IsGpr64(operands[0].reg.value)) {
            return TranslateResult::kUnimplemented;
        }
        const Arm64Reg src = ZydisGprToArm64(operands[0].reg.value);
        // Str64PreIndex takes the magnitude; imm=8 → #-8 in encoding.
        buf.Str64PreIndex(src, X4, 8);
        return TranslateResult::kTranslated;
    }

    // ── POP ──
    // POP reg: LDR Xreg, [X4], #8 (load, then post-increment RSP).
    TranslateResult TranslatePop(CodeBuffer& buf,
                                    const ZydisDecodedOperand* operands) {
        if (operands[0].type != ZYDIS_OPERAND_TYPE_REGISTER ||
            !IsGpr64(operands[0].reg.value)) {
            return TranslateResult::kUnimplemented;
        }
        const Arm64Reg dst = ZydisGprToArm64(operands[0].reg.value);
        buf.Ldr64PostIndex(dst, X4, 8);
        return TranslateResult::kTranslated;
    }

    // ── INC reg (dst = dst + 1) ──
    TranslateResult TranslateInc(CodeBuffer& buf,
                                    const ZydisDecodedInstruction& inst,
                                    const ZydisDecodedOperand* operands) {
        if (operands[0].type != ZYDIS_OPERAND_TYPE_REGISTER ||
            !IsGpr64(operands[0].reg.value)) {
            return TranslateResult::kUnimplemented;
        }
        const Arm64Reg dst = ZydisGprToArm64(operands[0].reg.value);
        buf.AddImm(dst, dst, 1);
        // INC sets OF/SF/ZF/PF (but not CF). For the MVP we skip the flag
        // update — the interpreter handles this when needed.
        return TranslateResult::kTranslated;
    }

    // ── DEC reg (dst = dst - 1) ──
    TranslateResult TranslateDec(CodeBuffer& buf,
                                    const ZydisDecodedInstruction& inst,
                                    const ZydisDecodedOperand* operands) {
        if (operands[0].type != ZYDIS_OPERAND_TYPE_REGISTER ||
            !IsGpr64(operands[0].reg.value)) {
            return TranslateResult::kUnimplemented;
        }
        const Arm64Reg dst = ZydisGprToArm64(operands[0].reg.value);
        buf.SubImm(dst, dst, 1);
        return TranslateResult::kTranslated;
    }

    // ── NEG reg (dst = -src; sets all arithmetic flags) ──
    TranslateResult TranslateNeg(CodeBuffer& buf,
                                    const ZydisDecodedInstruction& inst,
                                    const ZydisDecodedOperand* operands) {
        if (operands[0].type != ZYDIS_OPERAND_TYPE_REGISTER ||
            !IsGpr64(operands[0].reg.value)) {
            return TranslateResult::kUnimplemented;
        }
        const Arm64Reg dst = ZydisGprToArm64(operands[0].reg.value);
        buf.Neg(dst, dst);
        // NEG is equivalent to 0 - src; we'd need SUBS to set flags and
        // convert. For the MVP we skip the flag update.
        return TranslateResult::kTranslated;
    }

    // ── NOT reg (dst = ~src; no flag changes) ──
    TranslateResult TranslateNot(CodeBuffer& buf,
                                    const ZydisDecodedInstruction& inst,
                                    const ZydisDecodedOperand* operands) {
        if (operands[0].type != ZYDIS_OPERAND_TYPE_REGISTER ||
            !IsGpr64(operands[0].reg.value)) {
            return TranslateResult::kUnimplemented;
        }
        const Arm64Reg dst = ZydisGprToArm64(operands[0].reg.value);
        // NOT Xd, Xn = EOR Xd, Xn, #-1 = MVN Xd, Xn.
        // We don't have an immediate-form EOR; materialise -1 in X21.
        buf.MovImm64(X21, ~0ULL);
        buf.Eor(dst, dst, X21);
        return TranslateResult::kTranslated;
    }

    // ── MOVSX / MOVSXD (sign-extend) ──
    TranslateResult TranslateMovsx(CodeBuffer& buf,
                                      const ZydisDecodedInstruction& inst,
                                      const ZydisDecodedOperand* operands) {
        if (operands[0].type != ZYDIS_OPERAND_TYPE_REGISTER ||
            operands[1].type != ZYDIS_OPERAND_TYPE_REGISTER) {
            return TranslateResult::kUnimplemented;
        }
        if (!IsGpr64(operands[0].reg.value)) {
            return TranslateResult::kUnimplemented;
        }
        // Source can be 8/16/32-bit. For 32-bit source, use SBFM (alias ASR).
        // For the MVP, handle only 32-bit source (MOVSXD).
        if (operands[1].reg.value >= ZYDIS_REGISTER_EAX &&
            operands[1].reg.value <= ZYDIS_REGISTER_R15D) {
            const Arm64Reg dst = ZydisGprToArm64(operands[0].reg.value);
            const Arm64Reg src = ZydisGprToArm64(
                ZydisRegister(ZYDIS_REGISTER_RAX + (operands[1].reg.value - ZYDIS_REGISTER_EAX)));
            // SBFM Xd, Xn, #0, #31 — sign-extend 32-bit value to 64-bit.
            // Encoding (SBFM 64-bit): 0xD3400000 base, but with opc=00 (SBFM):
            //   bits 31=1, 30-29=00, 28-23=100110, 22=N=1, 21-16=immr, 15-10=imms, 9-5=Rn, 4-0=Rd
            //   Base: 0xD3400000 — wait, that's UBFM. SBFM base: 0xD3000000? Let me re-check.
            //   Actually: UBFM 64-bit base = 0xD3400000, SBFM 64-bit base = 0xD3000000.
            buf.Emit32(0xD3000000 | (0 << 16) | (31 << 10) | ((u32)src << 5) | (u32)dst);
            return TranslateResult::kTranslated;
        }
        // For 8/16-bit sources, fall back (would need SXTB/SXTH which
        // aren't in our emitter).
        return TranslateResult::kUnimplemented;
    }

    // ── MOVZX (zero-extend) ──
    TranslateResult TranslateMovzx(CodeBuffer& buf,
                                       const ZydisDecodedInstruction& inst,
                                       const ZydisDecodedOperand* operands) {
        if (operands[0].type != ZYDIS_OPERAND_TYPE_REGISTER ||
            operands[1].type != ZYDIS_OPERAND_TYPE_REGISTER) {
            return TranslateResult::kUnimplemented;
        }
        if (!IsGpr64(operands[0].reg.value)) {
            return TranslateResult::kUnimplemented;
        }
        // Source can be 8/16/32-bit. For 32-bit source, use UBFM (alias LSR).
        if (operands[1].reg.value >= ZYDIS_REGISTER_EAX &&
            operands[1].reg.value <= ZYDIS_REGISTER_R15D) {
            const Arm64Reg dst = ZydisGprToArm64(operands[0].reg.value);
            const Arm64Reg src = ZydisGprToArm64(
                ZydisRegister(ZYDIS_REGISTER_RAX + (operands[1].reg.value - ZYDIS_REGISTER_EAX)));
            // UBFM Xd, Xn, #0, #31 — zero-extend 32-bit value to 64-bit.
            buf.Ubfx(dst, src, 0, 32);
            return TranslateResult::kTranslated;
        }
        // For 8/16-bit sources, fall back (would need UXTB/UXTH).
        return TranslateResult::kUnimplemented;
    }

    // ── CMOV (conditional move) ──
    // CMOVcc dst, src: if condition is true, dst = src.
    // We emit: test condition → CBNZ skip → MOV dst, src → skip:
    TranslateResult TranslateCmov(CodeBuffer& buf, ZydisMnemonic mn,
                                    const ZydisDecodedInstruction& inst,
                                    const ZydisDecodedOperand* operands) {
        if (operands[0].type != ZYDIS_OPERAND_TYPE_REGISTER ||
            operands[1].type != ZYDIS_OPERAND_TYPE_REGISTER) {
            return TranslateResult::kUnimplemented;
        }
        if (!IsGpr64(operands[0].reg.value) ||
            !IsGpr64(operands[1].reg.value)) {
            return TranslateResult::kUnimplemented;
        }
        const Arm64Reg dst = ZydisGprToArm64(operands[0].reg.value);
        const Arm64Reg src = ZydisGprToArm64(operands[1].reg.value);

        // Emit the condition test (puts 1/0 in X21).
        const bool inverted = EmitCmovTest(buf, mn);
        // CMOV is "move if condition true", so:
        //   if (cond) dst = src;
        // We emit: B.cond skip (if cond is true, skip the no-move path).
        // For simplicity: branch if NOT cond (i.e., skip the MOV).
        const s32 branch_offset = buf.current_offset();
        if (inverted) {
            // If "flag == 0" means cond true, branch over the MOV when X21 != 0.
            buf.Cbnz(X21, 0);  // placeholder
        } else {
            // If "flag == 1" means cond true, branch over the MOV when X21 == 0.
            buf.Cbz(X21, 0);   // placeholder
        }
        // MOV dst, src (executed if condition false).
        buf.Mov(dst, src);
        // skip:
        const s32 skip_offset = buf.current_offset();
        // Patch the CBZ/CBNZ with the offset to skip.
        const s32 cb_offset = skip_offset - branch_offset;
        const u32 cb_base = inverted ? 0xB5000000u : 0xB4000000u;
        const u32 cb_insn = cb_base | (((u32)(cb_offset / 4) & 0x7FFFFu) << 5) | (u32)X21;
        u8* base = const_cast<u8*>(buf.data());
        base[branch_offset + 0] = (u8)(cb_insn & 0xFF);
        base[branch_offset + 1] = (u8)((cb_insn >> 8) & 0xFF);
        base[branch_offset + 2] = (u8)((cb_insn >> 16) & 0xFF);
        base[branch_offset + 3] = (u8)((cb_insn >> 24) & 0xFF);
        return TranslateResult::kTranslated;
    }

    // Emit the condition test for a CMOV. Returns true if "flag == 0" means
    // the condition is true (so CBNZ skips the move); false if "flag == 1"
    // means the condition is true (so CBZ skips the move).
    bool EmitCmovTest(CodeBuffer& buf, ZydisMnemonic mn) {
        // Map CMOVcc to the same Jcc condition test (the underlying
        // condition is the same).
        switch (mn) {
        case ZYDIS_MNEMONIC_CMOVO:  return EmitJccTest(buf, ZYDIS_MNEMONIC_JO);
        case ZYDIS_MNEMONIC_CMOVNO: return EmitJccTest(buf, ZYDIS_MNEMONIC_JNO);
        case ZYDIS_MNEMONIC_CMOVB:  return EmitJccTest(buf, ZYDIS_MNEMONIC_JB);
        case ZYDIS_MNEMONIC_CMOVNB:return EmitJccTest(buf, ZYDIS_MNEMONIC_JNB);
        case ZYDIS_MNEMONIC_CMOVZ:  return EmitJccTest(buf, ZYDIS_MNEMONIC_JZ);
        case ZYDIS_MNEMONIC_CMOVNZ:return EmitJccTest(buf, ZYDIS_MNEMONIC_JNZ);
        case ZYDIS_MNEMONIC_CMOVBE:return EmitJccTest(buf, ZYDIS_MNEMONIC_JBE);
        case ZYDIS_MNEMONIC_CMOVNBE:return EmitJccTest(buf, ZYDIS_MNEMONIC_JNBE);
        case ZYDIS_MNEMONIC_CMOVS: return EmitJccTest(buf, ZYDIS_MNEMONIC_JS);
        case ZYDIS_MNEMONIC_CMOVNS:return EmitJccTest(buf, ZYDIS_MNEMONIC_JNS);
        case ZYDIS_MNEMONIC_CMOVP: return EmitJccTest(buf, ZYDIS_MNEMONIC_JP);
        case ZYDIS_MNEMONIC_CMOVNP:return EmitJccTest(buf, ZYDIS_MNEMONIC_JNP);
        case ZYDIS_MNEMONIC_CMOVL: return EmitJccTest(buf, ZYDIS_MNEMONIC_JL);
        case ZYDIS_MNEMONIC_CMOVNL:return EmitJccTest(buf, ZYDIS_MNEMONIC_JNL);
        case ZYDIS_MNEMONIC_CMOVLE:return EmitJccTest(buf, ZYDIS_MNEMONIC_JLE);
        case ZYDIS_MNEMONIC_CMOVNLE:return EmitJccTest(buf, ZYDIS_MNEMONIC_JNLE);
        default:
            buf.MovImm64(X21, 0);
            return false;
        }
    }

    // ── SHL / SHR / SAR reg, imm ──
    TranslateResult TranslateShift(CodeBuffer& buf,
                                       const ZydisDecodedInstruction& inst,
                                       const ZydisDecodedOperand* operands) {
        if (operands[0].type != ZYDIS_OPERAND_TYPE_REGISTER ||
            operands[1].type != ZYDIS_OPERAND_TYPE_IMMEDIATE) {
            // reg,cl form — fall back for the MVP.
            return TranslateResult::kUnimplemented;
        }
        if (!IsGpr64(operands[0].reg.value)) {
            return TranslateResult::kUnimplemented;
        }
        const Arm64Reg dst = ZydisGprToArm64(operands[0].reg.value);
        const u64 shift = operands[1].imm.value.u & 63;

        switch (inst.mnemonic) {
        case ZYDIS_MNEMONIC_SHL:
            buf.LslImm(dst, dst, (u32)shift);
            return TranslateResult::kTranslated;
        case ZYDIS_MNEMONIC_SHR:
            buf.LsrImm(dst, dst, (u32)shift);
            return TranslateResult::kTranslated;
        case ZYDIS_MNEMONIC_SAR:
            // SAR Xd, Xn, #shift = SBFM Xd, Xn, #shift, #63.
            // Encoding (SBFM 64-bit): 0xD3000000 | (shift << 16) | (63 << 10) | (Xn << 5) | Xd
            buf.Emit32(0xD3000000u | ((u32)shift << 16) | (63u << 10) |
                       ((u32)dst << 5) | (u32)dst);
            return TranslateResult::kTranslated;
        default:
            return TranslateResult::kUnimplemented;
        }
    }

    // =====================================================================
    //  Memory operand handling
    // =====================================================================

    // Compute the effective address of a memory operand into `out_reg`
    // (typically X20). Returns true on success.
    bool EmitLeaMem(CodeBuffer& buf, Arm64Reg out_reg,
                    const ZydisDecodedOperand& mem, u64 rip, u64 next_rip) {
        if (mem.type != ZYDIS_OPERAND_TYPE_MEMORY) {
            return false;
        }

        const bool has_base  = mem.mem.base  != ZYDIS_REGISTER_NONE;
        const bool has_index = mem.mem.index != ZYDIS_REGISTER_NONE;
        const s64 disp = mem.mem.disp.value;
        const u8 scale = mem.mem.scale ? mem.mem.scale : 1;

        // RIP-relative addressing: [RIP + disp].
        // The address is (X16 + (next_rip - block_start_rip) + disp) — but
        // since X16 may not track current RIP continuously, we compute
        // the absolute address at translation time:
        //   address = next_rip + disp
        // and materialise it directly with MovImm64.
        if (mem.mem.base == ZYDIS_REGISTER_RIP) {
            const u64 addr = next_rip + disp;
            buf.MovImm64(out_reg, addr);
            return true;
        }

        // [base] only.
        if (has_base && !has_index) {
            if (!IsGpr64(mem.mem.base)) {
                return false;
            }
            const Arm64Reg base = ZydisGprToArm64(mem.mem.base);
            if (disp == 0) {
                buf.Mov(out_reg, base);
                return true;
            }
            if (disp > 0 && disp <= 0xFFF) {
                buf.AddImm(out_reg, base, (u32)disp);
                return true;
            }
            if (disp < 0 && -disp <= 0xFFF) {
                buf.SubImm(out_reg, base, (u32)(-disp));
                return true;
            }
            // Large displacement: materialise in X21 and add.
            buf.MovImm64(X21, (u64)disp);
            buf.Add(out_reg, base, X21);
            return true;
        }

        // [index] only (no base).
        if (!has_base && has_index) {
            if (!IsGpr64(mem.mem.index)) {
                return false;
            }
            const Arm64Reg idx = ZydisGprToArm64(mem.mem.index);
            if (scale == 1) {
                buf.Mov(out_reg, idx);
            } else {
                buf.Movz16(X21, scale);
                buf.Madd(out_reg, idx, X21, Arm64Reg::ZR);
            }
            if (disp != 0) {
                if (disp > 0 && disp <= 0xFFF) {
                    buf.AddImm(out_reg, out_reg, (u32)disp);
                } else if (disp < 0 && -disp <= 0xFFF) {
                    buf.SubImm(out_reg, out_reg, (u32)(-disp));
                } else {
                    buf.MovImm64(X21, (u64)disp);
                    buf.Add(out_reg, out_reg, X21);
                }
            }
            return true;
        }

        // [base + index*scale + disp].
        if (has_base && has_index) {
            if (!IsGpr64(mem.mem.base) || !IsGpr64(mem.mem.index)) {
                return false;
            }
            const Arm64Reg base = ZydisGprToArm64(mem.mem.base);
            const Arm64Reg idx  = ZydisGprToArm64(mem.mem.index);
            // MADD out = base + index * scale.
            if (scale == 1) {
                buf.Add(out_reg, base, idx);
            } else {
                buf.Movz16(X21, scale);
                buf.Madd(out_reg, idx, X21, base);
            }
            if (disp != 0) {
                if (disp > 0 && disp <= 0xFFF) {
                    buf.AddImm(out_reg, out_reg, (u32)disp);
                } else if (disp < 0 && -disp <= 0xFFF) {
                    buf.SubImm(out_reg, out_reg, (u32)(-disp));
                } else {
                    buf.MovImm64(X21, (u64)disp);
                    buf.Add(out_reg, out_reg, X21);
                }
            }
            return true;
        }

        // [disp] — absolute address (no base, no index).
        if (!has_base && !has_index) {
            buf.MovImm64(out_reg, (u64)disp);
            return true;
        }

        return false;
    }

    // Emit a 64-bit load from a memory operand into `dst`.
    // `rip` and `next_rip` are the guest RIPs of the current instruction
    // and the next instruction (next_rip = rip + inst.length), used for
    // RIP-relative addressing.
    TranslateResult EmitLoadMem(CodeBuffer& buf, Arm64Reg dst,
                                 const ZydisDecodedOperand& mem,
                                 u64 rip, u64 next_rip) {
        // Compute address into X20.
        if (!EmitLeaMem(buf, X20, mem, rip, next_rip)) {
            return TranslateResult::kUnimplemented;
        }
        // LDR dst, [X20]
        buf.Ldr64(dst, X20, 0);
        return TranslateResult::kTranslated;
    }

    // Emit a 64-bit store of `src` into a memory operand.
    TranslateResult EmitStoreMem(CodeBuffer& buf, Arm64Reg src,
                                  const ZydisDecodedOperand& mem,
                                  u64 rip, u64 next_rip) {
        // Compute address into X20 (using X21 as a temp).
        if (!EmitLeaMem(buf, X20, mem, rip, next_rip)) {
            return TranslateResult::kUnimplemented;
        }
        // STR src, [X20]
        buf.Str64(src, X20, 0);
        return TranslateResult::kTranslated;
    }

    // =====================================================================
    //  Flag conversion
    // =====================================================================

    // Convert ARM64 NZCV (in X20) to x86 RFLAGS format and store in X17.
    // The NZCV bits are at positions 31 (N), 30 (Z), 29 (C), 28 (V).
    // The x86 RFLAGS bits we set are: SF (7), ZF (6), CF (0), OF (11).
    // For SUB/CMP, the ARM64 C bit is inverted (ARM64 C=1 means no borrow,
    // x86 CF=1 means borrow) — pass is_sub=true to invert.
    //
    // We overwrite X17 entirely (PF, AF, IF, DF, etc. are reset) — the
    // interpreter can compute those when needed. Uses X20, X21, X22.
    void EmitConvertFlags(CodeBuffer& buf, bool is_sub) {
        // MRS X20, NZCV — read NZCV into X20 (bits 31-28 hold N/Z/C/V).
        buf.MrsNzcv(X20);

        // SF: extract N (bit 31), shift to bit 7. MOV X17 = SF (overwrite).
        buf.Ubfx(X21, X20, 31, 1);
        buf.LslImm(X17, X21, 7);

        // ZF: extract Z (bit 30), shift to bit 6. ORR into X17.
        buf.Ubfx(X21, X20, 30, 1);
        buf.LslImm(X22, X21, 6);
        buf.Orr(X17, X17, X22);

        // CF: extract C (bit 29). For SUB/CMP, invert.
        buf.Ubfx(X21, X20, 29, 1);
        if (is_sub) {
            // EOR X21, X21, #1 → invert. We materialise 1 in X22.
            buf.Movz16(X22, 1);
            buf.Eor(X21, X21, X22);
        }
        buf.Orr(X17, X17, X21);

        // OF: extract V (bit 28), shift to bit 11. ORR into X17.
        buf.Ubfx(X21, X20, 28, 1);
        buf.LslImm(X22, X21, 11);
        buf.Orr(X17, X17, X22);
    }

    // =====================================================================
    //  Helpers
    // =====================================================================

    // Emit MOV X16, #imm (using MovImm64). Used for block terminators.
    void EmitSetX16Imm(CodeBuffer& buf, u64 imm) {
        buf.MovImm64(X16, imm);
    }

    // Emit RET (returns to the trampoline exit handler via X30).
    void buf_Ret(CodeBuffer& buf) {
        buf.Ret();
    }
};

// ─────────────────────────────────────────────────────────────────────────
//  Aarch64JitBackend public interface
// ─────────────────────────────────────────────────────────────────────────

Aarch64JitBackend::Aarch64JitBackend()
    : m_impl(std::make_unique<Impl>()) {
    if (m_impl->run_block) {
        LOG_INFO(Core_Cpu, "ARM64 JIT backend initialized (trampoline ready)");
    } else {
        LOG_WARNING(Core_Cpu, "ARM64 JIT backend initialized without trampoline; "
                    "all execution will fall back to the interpreter");
    }
}

Aarch64JitBackend::~Aarch64JitBackend() = default;

u64 Aarch64JitBackend::Execute(u64 rip, const GuestCallContext& ctx) {
    // If the trampoline failed to initialise, fall back entirely to the
    // interpreter — we can't execute translated code without it.
    if (!m_impl->run_block) {
        static X64InterpreterBackend s_interpreter;
        return s_interpreter.Execute(rip, ctx);
    }

    // Main loop: look up (or translate) the block for `rip`, run it via
    // the trampoline, and handle the fallback path when a block exits at
    // an unimplemented instruction.
    while (true) {
        const BlockEntry* block = m_impl->block_cache.Lookup(rip);
        if (!block) {
            // Translate the block. If translation fails entirely (e.g.,
            // out of code-cache space, or guest_rip is unmapped), fall
            // back to the interpreter for this whole block.
            void* host_code = m_impl->TranslateBlock(rip);
            if (!host_code) {
                static X64InterpreterBackend s_interpreter;
                return s_interpreter.Execute(rip, ctx);
            }
            block = m_impl->block_cache.Lookup(rip);
            if (!block) {
                static X64InterpreterBackend s_interpreter;
                return s_interpreter.Execute(rip, ctx);
            }
        }

        // Set up the shared state with the current RIP and run the block.
        m_impl->state.rip = rip;
        const u64 new_rip = m_impl->run_block(&m_impl->state, block->host_code);

        // Check if the block exited at an unimplemented instruction.
        // If so, dispatch the interpreter for that single instruction
        // (the interpreter will run to the next block terminator or to
        // its own exit condition). After the interpreter returns, re-
        // enter the JIT at the new RIP.
        auto fb = m_impl->fallback_rips.find(rip);
        if (fb != m_impl->fallback_rips.end() && fb->second == new_rip) {
            static X64InterpreterBackend s_interpreter;
            // The interpreter creates its own internal state; for the MVP
            // we don't synchronise GPRs between the JIT and the interpreter
            // across the fallback boundary. (A full implementation would
            // either share the X64CpuState or copy GPRs into the
            // interpreter's state before calling.)
            const u64 cont = s_interpreter.Execute(new_rip, ctx);
            rip = cont;
            continue;
        }

        // If the new RIP is the same as the previous one and the block
        // doesn't have a fallback, we'd loop forever — break out so the
        // runtime can decide what to do. (In practice the runtime will
        // call Execute() again with a fresh RIP from the host.)
        if (new_rip == rip) {
            return new_rip;
        }

        rip = new_rip;
    }
}

void Aarch64JitBackend::SetRuntimeConfig(const std::string& key, const std::string& value) {
    // Runtime config for the JIT (e.g., cache size, trace mode).
    // For the MVP, this is a no-op.
    (void)key;
    (void)value;
}

} // namespace Core::Cpu
