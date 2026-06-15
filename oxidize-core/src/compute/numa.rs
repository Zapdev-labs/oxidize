//! NUMA weight replication helpers.
//!
//! Full replication is optional and configured at model load on Linux dual-socket
//! hosts. When replication is not active, [`local_slice`] returns the input slice.

pub fn local_slice<T>(slice: &[T]) -> &[T] {
    slice
}
