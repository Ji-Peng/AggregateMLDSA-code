//! Two reusable spare tables shared by the live groups of a prover driver.
//!
//! Each group owns its current table. A fold checks out a spare and returns its
//! previous table after all readers finish; terminal tables return when groups drop.

use std::sync::{Arc, Mutex};
use crate::OpeningField as E;

pub(crate) type SharedArena = Arc<Mutex<TableArena>>;

#[derive(Default)]
pub(crate) struct TableArena {
    a: Vec<E>,
    b: Vec<E>,
}

impl TableArena {
    pub(crate) fn shared() -> SharedArena {
        Arc::new(Mutex::new(Self::default()))
    }

    pub(crate) fn take(&mut self, len: usize) -> Vec<std::mem::MaybeUninit<E>> {
        // Use the smallest fitting spare, preserving the larger one for other groups.
        let slot = if self.b.capacity() >= len { &mut self.b } else { &mut self.a };
        let mut table = std::mem::take(slot);
        table.clear();
        if table.capacity() < len && std::env::var_os("AGG_ALLOC_TRACE").is_some() {
            eprintln!("arena-grow bytes={} previous_capacity={}", len * std::mem::size_of::<E>(), table.capacity());
        }
        super::buf::reuse(table, len)
    }

    pub(crate) fn put(&mut self, mut table: Vec<E>) {
        table.clear();
        if table.capacity() > self.a.capacity() {
            std::mem::swap(&mut table, &mut self.a);
        }
        if table.capacity() > self.b.capacity() {
            std::mem::swap(&mut table, &mut self.b);
        }
    }
}
