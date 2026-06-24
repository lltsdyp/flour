//! Distributed KV Cache data-management layer.
//!
//! Master owns metadata (worker liveness, capacity, object state, placement,
//! leases, route selection). Workers own KV bytes and serve them directly to
//! requesters. See `docs/plan/plan-distkv.md` for the full specification.

pub mod client;
pub mod http;
pub mod master;
pub mod protocol;
pub mod registration;
pub mod scheduler;
pub mod test;
pub mod worker;
