//! The safe entry points refuse inputs shorter than their vector kernels read.
use agg_simd::{
    fused::{eval_items, fused_pass, Fill, Terms, MAX_SLOT_TERMS},
    pattern::{fill_rows, Bottom, Columns, Leaves},
    K,
};
use jolt_field::Ring;

/// A fold whose input lane holds fewer than two rows per output row.
#[test]
#[should_panic(expected = "input lanes cover the folded rows")]
fn fold_of_an_undersized_lane_panics() {
    let rows = 64;
    let mut out = vec![0_u64; 2 * rows];
    let short = vec![0_u64; 2 * rows - 8];
    let full = vec![0_u64; 2 * rows];
    let lanes = [(&short[..], &full[..])];
    let _ = fused_pass::<Columns>(Some((&mut out, 1, rows)), &[], Some(Fill::Fold(K::from_u64(3), &lanes)), None);
}

/// A pattern fill over fewer than four cells per output row.
#[test]
#[should_panic(expected = "leaves cover the rows")]
fn pattern_fill_of_too_few_cells_panics() {
    let rows = 64;
    let cells = vec![0_i32; 4 * rows - 4];
    let columns = Columns { lanes: vec![&cells, &cells, &cells, &cells] };
    let mut out = vec![0_u64; 2 * 4 * rows];
    let _ = fused_pass(Some((&mut out, 4, rows)), &[], Some(Fill::Pattern([K::from_u64(1); 4], &columns)), None);
}

/// Column lanes of different lengths.
#[test]
#[should_panic(expected = "column lanes of equal length")]
fn columns_of_unequal_lanes_panic() {
    let (a, b) = (vec![0_i32; 64], vec![0_i32; 60]);
    let _ = Columns { lanes: vec![&a, &b] }.rows();
}

/// More column lanes than the four a cell holds.
#[test]
#[should_panic(expected = "one to four column lanes")]
fn five_column_lanes_panic() {
    let a = vec![0_i32; 64];
    let _ = Columns { lanes: vec![&a; 5] }.rows();
}

/// The two halves of a stack bottom of different lengths.
#[test]
#[should_panic(expected = "the two halves of the column")]
fn bottom_of_unequal_halves_panics() {
    let (a, b) = (vec![0_i32; 64], vec![0_i32; 32]);
    let rows = 8;
    let mut out = vec![0_u64; 2 * 2 * rows];
    let base = out.as_mut_ptr() as usize;
    let lanes = [(base, base + 8 * rows), (base + 16 * rows, base + 24 * rows)];
    // SAFETY: each output address holds `rows` writable words of `out`.
    unsafe { fill_rows(&Bottom { v0: &a, v1: &b }, [K::from_u64(1); 4], &lanes, 0, rows) };
}

/// A slot with more terms than one unreduced accumulator takes.
#[test]
#[should_panic(expected = "at most MAX_SLOT_TERMS terms per slot")]
fn a_slot_over_the_term_bound_panics() {
    let lane = vec![0_u64; 32];
    let lanes = [(&lane[..], &lane[..])];
    let linear = vec![(0_usize, 0_usize); MAX_SLOT_TERMS + 1];
    let terms = Terms { linear: &linear, products: &[], slots: 1 };
    let mut out = [[K::from_u64(0); 3]; 1];
    eval_items(&lanes, terms, None, 0, 16, true, &mut out);
}
