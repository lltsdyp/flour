use axum::extract::State;
use axum::Json;

use super::openai::{ModelObject, ModelsListResponse};
use super::AppState;

pub async fn list_models(State(state): State<AppState>) -> Json<ModelsListResponse> {
    let engine = state.engine.lock().unwrap();
    Json(ModelsListResponse {
        object: "list".into(),
        data: vec![ModelObject {
            id: engine.model_id().to_string(),
            object: "model".into(),
            created: state.started_at,
            owned_by: "flour".into(),
        }],
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::AppState;
    use crate::engine::Engine;
    use axum::extract::State;
    use std::sync::{Arc, Mutex};

    fn fixture_engine_dir() -> tempfile::TempDir {
        crate::engine::tests::fixture_dir_for_external_use()
    }

    #[tokio::test]
    async fn lists_the_single_loaded_model() {
        let dir = fixture_engine_dir();
        let engine = Engine::load(dir.path()).unwrap();
        let expected_id = engine.model_id().to_string();
        let state = AppState { engine: Arc::new(Mutex::new(engine)), started_at: 0 };

        let Json(resp) = list_models(State(state)).await;
        assert_eq!(resp.object, "list");
        assert_eq!(resp.data.len(), 1);
        assert_eq!(resp.data[0].id, expected_id);
        assert_eq!(resp.data[0].object, "model");
    }
}
