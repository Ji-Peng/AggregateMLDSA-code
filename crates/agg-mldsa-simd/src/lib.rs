//! Vectorized kernels and SIMD primitives for ML-DSA signature aggregation.

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(clippy::undocumented_unsafe_blocks)]

pub mod dispatch;
pub mod finish;
pub mod hash;
pub mod isa;
pub mod keccak;
pub mod ntt;
pub mod params;
pub mod poly;
pub mod sample;

pub use dispatch::{available, engine, engine_for, parse_backend, Backend, Engine};
pub use finish::{FinishError, FinishOut, FinishRow, FinishRow3P, FinishRowExact, HuCols};
pub use keccak::{Sink, Xof, XofJob};
pub use ntt::{Form, Scale};
pub use params::{
    consts, ParamSet, Prime, PrimeConsts, Set44, Set65, Set87, CRT_C, CRT_C_P3, P2, P3, Q, QINV32,
};
pub use poly::{Block, Challenge, Challenge39, ExpandedA, Poly32, RejState};
