//! Coroutine and async operations.
//!
//! This module implements the opcode handlers for Python's async/await
//! feature set (PEP 492/525/530):
//!
//! - `GetAwaitable` - Convert object to awaitable for `await`
//! - `GetAIter` - Get async iterator via `__aiter__`
//! - `GetANext` - Get next value via `__anext__`
//! - `EndAsyncFor` - Handle StopAsyncIteration
//! - `Send` - Send value to coroutine/generator
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                     Coroutine Type Hierarchy                     │
//! ├─────────────────────────────────────────────────────────────────┤
//! │                                                                  │
//! │  Native Coroutine     Generator Coroutine    Custom Awaitable  │
//! │  (async def)          (@types.coroutine)     (__await__ method) │
//! │        │                      │                     │           │
//! │        └──────────────────────┴─────────────────────┘           │
//! │                              │                                   │
//! │                       GetAwaitable                               │
//! │                              │                                   │
//! │                        YieldFrom                                 │
//! │                              │                                   │
//! │                      Execution Loop                              │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Performance
//!
//! All operations are optimized for the common case:
//! - Native coroutines: O(1) type check, zero allocation
//! - __await__ lookup: Cached via method cache
//! - Async iteration: Amortized O(1) per element

mod end_async_for;
mod get_aiter;
mod get_anext;
mod get_awaitable;
mod protocol;
mod send;

#[cfg(test)]
mod tests;

pub use end_async_for::end_async_for;
pub use get_aiter::get_aiter;
pub use get_anext::get_anext;
pub use get_awaitable::get_awaitable;
pub use send::send;
