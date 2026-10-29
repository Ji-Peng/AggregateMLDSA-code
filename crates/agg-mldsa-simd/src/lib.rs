//! Vectorized kernels and SIMD primitives for ML-DSA signature aggregation.

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(clippy::undocumented_unsafe_blocks)]

pub mod dispatch;
pub mod finish;
pub mod hash;
pub mod keccak;
pub mod ntt;
pub mod params;
pub mod poly;
pub mod sample;

pub use dispatch::{available, engine, parse_backend, Backend, Engine};
pub use finish::{FinishError, FinishOut, FinishRow, HuCols};
pub use keccak::{Sink, Xof, XofJob};
pub use ntt::{Form, Scale};
pub use params::{consts, Prime, PrimeConsts, CRT_C, QINV32};
pub use poly::{Block, Challenge39, ExpandedA, Poly32, RejState};
