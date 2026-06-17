pub mod error;
pub mod models;
pub mod openai;

use std::sync::{Arc, Mutex};

use crate::engine::Engine;

#[derive(Clone)]
pub struct AppState {
    pub engine: Arc<Mutex<Engine>>,
    pub started_at: u64,
}
