//! Runtime-dispatched arithmetic for the fixed Solinas field and its quadratic extension.
#![deny(unsafe_op_in_unsafe_fn)]
pub mod buf;
pub mod cast;
pub mod detect;
pub mod scalar;
pub type F = jolt_field::Fp64<18446744073709551557>;
pub type K = jolt_field::Ext2<F>;
pub const P: u64 = 18446744073709551557;
pub mod dot;
pub mod fold;
pub mod fp;
pub mod k;
#[cfg(target_arch = "x86_64")]
mod x86;
#[doc(hidden)]
pub mod wide;
pub mod negacyclic;
pub mod lookup;
pub mod fused;
pub mod mask;
pub mod pattern;
pub mod eval;
pub mod digits;
