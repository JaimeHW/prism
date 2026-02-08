<p align="center">
  <h1 align="center">⬡ Prism</h1>
  <p align="center"><i>A high-performance Python runtime with JIT compilation</i></p>
</p>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#building">Building</a> •
  <a href="#project-status">Status</a> •
  <a href="#contributing">Contributing</a>
</p>

---

Prism is a from-scratch implementation of the Python 3.12 runtime, engineered for performance through a custom multi-tier JIT compiler. Written entirely in Rust, Prism combines a register-based bytecode interpreter with an optimizing compiler that generates native x64 machine code.

## Features

### Multi-Tier Execution Engine
- **Tier 0 Interpreter** — Register-based bytecode VM with static dispatch tables and arithmetic fast-paths
- **Tier 1 Template JIT** — Direct bytecode-to-machine-code translation with inline caching, type specialization, and deoptimization IC
- **Tier 2 Optimizing JIT** — Sea-of-Nodes IR with aggressive optimizations and profile-guided compilation

### Advanced Optimizations
- **Inline Caching** — Monomorphic, polymorphic, and megamorphic caches for property access, method calls, and type dispatch
- **Type Speculation** — Profile-driven type guards with fast-path specialization for dict, list, and string operations
- **Profile-Guided Optimization** — Branch probability annotation and hot/cold code splitting from runtime profiles
- **On-Stack Replacement** — Mid-loop tier-up from interpreter to optimized code
- **Loop Optimizations** — LICM, Range Check Elimination, loop unrolling, and induction variable analysis
- **Function Inlining** — Budget-based graph merging with escape analysis
- **Strength Reduction** — Magic number division, multiplication decomposition into shifts/adds
- **Global Value Numbering** — Common subexpression elimination across the IR graph
- **Partial Redundancy Elimination** — Optimal code motion with PRE
- **Sparse Conditional Constant Propagation** — Aggressive constant folding with branch elimination
- **Instruction Combining** — Peephole optimizations on instruction sequences
- **Tail Call Optimization** — Stack frame reuse for tail-recursive functions
- **Dead Store Elimination** — Removal of stores that are never read
- **Auto-Vectorization** — SIMD transformation for compatible loops

### SIMD Acceleration
- **AVX2 Support** — 256-bit vector operations with YMM registers
- **AVX-512 Support** — 512-bit vector operations with ZMM registers and EVEX encoding
- **Opmask Predication** — Hardware-accelerated conditional vector operations (k0-k7)
- **Width-Aware Spill Slots** — Correct stack allocation for 128/256/512-bit vectors

### High-Performance Memory Management
- **Generational GC** — Immix-based heap with opportunistic evacuation and line-level marking
- **Thread-Local Allocation** — Zero-synchronization bump allocation via TLABs
- **Precise Stack Scanning** — Stackmap-driven root identification in JIT frames
- **Page-Protection Safepoints** — Minimal-overhead stop-the-world coordination

### V8-Style Object Model
- **Hidden Classes (Shapes)** — O(1) property access through inline slots and transition chains
- **NaN-Boxing** — Efficient 64-bit value representation for primitives and pointers
- **Small Integer Cache** — Pre-allocated integers from -5 to 256
- **Descriptor Protocol** — Full implementation of `@property`, `__slots__`, bound methods, and classmethod/staticmethod

### Zero-Cost Exception Handling
- **Table-Based Unwinding** — Zero runtime overhead for code paths that don't raise
- **Nested Handler Resolution** — Inner-first linear scan for try/except block matching
- **Exception Chaining** — Full `raise ... from ...` semantics with `__cause__` and `__context__`
- **Traceback System** — Complete stack trace generation with source coordinates
- **Flyweight Exception References** — Zero-copy exception propagation with inline caching
- **Exception State FSM** — Deterministic state machine for exception lifecycle
- **CPython 3.11+ Compatibility** — Full `PushExcInfo`/`PopExcInfo` semantics for `finally` blocks

### Generator Protocol
- **Stackless State Machines** — Minimal-overhead state capture and restoration
- **Liveness-Aware Frames** — Only live variables captured across yield points
- **Iterator Integration** — Full `__iter__`/`__next__` protocol support

### Structural Pattern Matching (PEP 634)
- **Decision Tree Compilation** — Maranget algorithm for optimal pattern dispatch
- **Full Pattern Support** — Literals, captures, wildcards, sequences, mappings, class patterns, OR/AS patterns
- **Guard Expressions** — Conditional pattern matching with arbitrary Python expressions

### I/O System
- **Layered Architecture** — `FileIO` → `BufferedIO` → `TextIO` stack
- **Buffer Pooling** — High-throughput file operations via reusable buffers
- **Thread-Safe Streams** — `sys.stdin`, `sys.stdout`, `sys.stderr` with locking

### Python 3.12 Compatibility
- **Complete Parser** — Pratt parser with 16 precedence tiers for Python's complex grammar
- **Scope Analysis** — Deep binding analysis with Local/Global/Cell/Free variable resolution
- **Arbitrary Precision Integers** — Full `BigInt` support for Python integer semantics
- **Standard Library** — `math`, `sys`, `os`, `time`, `re`, `json`, `collections`, `functools`, `itertools`, `io`, generators, and exception hierarchy modules

## Quick Start

Run a Python script:

```bash
prism script.py
```

Start the interactive REPL:

```bash
prism
```

## Architecture

Prism is organized as a modular Rust workspace:

```
prism/
├── prism_core      # Fundamental types: Value (NaN-boxing), Span, Error, Interning, Speculation
├── prism_parser    # Python 3.12 grammar and AST construction
├── prism_compiler  # Scope analysis, pattern matching, register-based bytecode emission
├── prism_vm        # Execution engine, interpreter, JIT bridge, stdlib modules
├── prism_jit       # Multi-tier JIT: IR, optimization passes, x64/ARM64 codegen, SIMD
├── prism_runtime   # Object system, shapes, types (list, dict, set, string, etc.)
├── prism_gc        # Generational Immix collector with TLABs and write barriers
├── prism_builtins  # Builtin function implementations
└── prism_cli       # Command-line interface
```

### Execution Pipeline

```
                              ┌─────────────────────────────────────────┐
                              │              Tier 2 JIT                 │
                              │  ┌─────────┐  ┌─────────┐  ┌─────────┐  │
                              │  │   GVN   │─▶│  LICM   │─▶│   RCE   │  │
                              │  └─────────┘  └─────────┘  └─────────┘  │
                              │         │                       │       │
                              │         ▼                       ▼       │
                              │  ┌─────────────────────────────────┐    │
                              │  │      Register Allocation        │    │
Source ─▶ Parser ─▶ Compiler ─┼─▶│    (Linear Scan + SIMD Regs)    │────┼──▶ Native x64
   │                          │  └─────────────────────────────────┘    │
   │                          └─────────────────▲───────────────────────┘
   │                                            │ OSR (hot loops)
   │                          ┌─────────────────┴───────────────────────┐
   │                          │           Tier 1 Template JIT           │
   │                          │  Bytecode → x64 · IC · Type Specialize  │
   │                          └─────────────────▲───────────────────────┘
   │                                            │ tier-up (hot functions)
   │                          ┌─────────────────┴───────────────────────┐
   └─────────────────────────▶│          Tier 0 Interpreter             │
                              │   Static dispatch · Type profiling      │
                              └─────────────────────────────────────────┘
```

### Optimization Pipeline

The Tier 2 JIT runs a 15-pass optimization pipeline across 6 phases:

| Phase | Pass | Description |
|:------|:-----|:------------|
| **Canonicalization** | SCCP | Sparse Conditional Constant Propagation |
| | Simplify | Algebraic identities, constant folding |
| | InstCombine | Peephole optimizations on instruction sequences |
| **Profile-Guided** | Branch Probability | Annotate branches with measured/estimated weights |
| | Hot/Cold Split | Partition code by execution temperature |
| **Local** | Copy Prop | Copy Propagation — eliminate redundant moves |
| | GVN | Global Value Numbering — CSE across the graph |
| | DSE | Dead Store Elimination |
| | PRE | Partial Redundancy Elimination |
| | Strength Reduce | Division → multiplication, mul → shift/add |
| **Loop** | LICM | Loop-Invariant Code Motion |
| | Unroll | Loop body replication to reduce overhead |
| | RCE | Range Check Elimination |
| **Interprocedural** | Inline | Budget-based function inlining |
| | Escape | Escape analysis for stack allocation |
| | Tail Call | Tail call optimization |
| **Cleanup** | DCE | Dead Code Elimination |

### JIT Tier Details

| Tier | Strategy | Trigger | Key Capabilities |
|:-----|:---------|:--------|:-----------------|
| **0** | Interpreter | Default | Inline caches, type feedback collection |
| **1** | Template | ~100 calls | Direct translation, IC fast paths, type specialization (dict/list/string) |
| **2** | Optimizing | ~1000 calls or hot loop | Full optimization pipeline, PGO, SIMD codegen |

### Object Model

Prism implements a V8-style hidden class system:

```
                    ┌──────────────────┐
                    │      Shape       │
                    │  (Hidden Class)  │
                    ├──────────────────┤
                    │ property: "x"    │────┐
                    │ slot: 0          │    │ transition
                    │ parent: ─────────┼────┘
                    └──────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
    ┌──────────────────┐            ┌──────────────────┐
    │     Object A     │            │     Object B     │
    ├──────────────────┤            ├──────────────────┤
    │ header (16 bytes)│            │ header (16 bytes)│
    │ slot[0]: 42      │            │ slot[0]: 100     │
    │ slot[1]: ...     │            │ slot[1]: ...     │
    └──────────────────┘            └──────────────────┘
```

Objects with identical property insertion order share the same Shape, enabling O(1) property access through fixed inline slots.

## Building

### Prerequisites

- **Rust 1.85+** (2024 Edition)
- **x64 architecture** (primary), ARM64 backend in progress

### Build

```bash
# Debug build
cargo build --workspace

# Release build (recommended for benchmarking)
cargo build --workspace --release

# Run tests
cargo test --workspace
```

### Release Profile

The release profile is tuned for maximum performance:

```toml
[profile.release]
lto = "fat"           # Link-time optimization
codegen-units = 1     # Single codegen unit for better optimization
panic = "abort"       # Reduced binary size
strip = true          # Strip symbols
```

## Project Status

Prism is under active development. Current status:

| Component | Status | Tests |
|:----------|:-------|------:|
| Parser | ✅ Complete | 167 |
| Compiler | ✅ Complete | 693 |
| Core Types & Values | ✅ Complete | 244 |
| VM & Interpreter | ✅ Complete | 3,927 |
| Object System (Shapes) | ✅ Complete | — |
| Descriptor Protocol | ✅ Complete | — |
| Pattern Matching | ✅ Complete | — |
| Exception Handling | ✅ Complete | — |
| JIT Tier 1 & 2 | ✅ Complete | 2,967 |
| SIMD Backend (AVX2/AVX-512) | ✅ Complete | — |
| Garbage Collector | ✅ Complete | 84 |
| Runtime Types | ✅ Complete | 789 |

**Total test coverage: 8,800+ tests**

### Roadmap

- [ ] Async/await coroutines
- [ ] ARM64 backend (infrastructure complete, codegen in progress)
- [ ] Extended standard library coverage
- [ ] Package import system

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT License ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.