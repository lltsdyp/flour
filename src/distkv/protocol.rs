//! Wire protocol types shared between Master, Worker, and Engine/Scheduler.
//!
//! Master owns metadata only; Workers own bytes. These structs are the
//! serialized contract over HTTP for both the metadata path (Master) and the
//! data path (Worker).

pub type WorkerId = String;
pub type ObjectKey = String;
pub type PutId = uuid::Uuid;
pub type LeaseId = uuid::Uuid;

/// Lifecycle state of an object as tracked by the Master.
///
/// `Absent` is represented by the absence of an entry, so it is intentionally
/// not part of this enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ObjectState {
    Writing,
    Complete,
    Failed,
    Removed,
}

/// Where a particular object generation lives.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Placement {
    pub worker_id: WorkerId,
    pub worker_epoch: u64,
    pub object_generation: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct RegisterRequest {
    pub worker_id: WorkerId,
    pub addr: String,
    pub capacity_bytes: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct RegisterResponse {
    pub epoch: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct HeartbeatRequest {
    pub worker_id: WorkerId,
    pub epoch: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct PutStartRequest {
    pub key: ObjectKey,
    pub size_bytes: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct PutStartResponse {
    pub put_id: PutId,
    pub worker_id: WorkerId,
    pub worker_addr: String,
    pub object_generation: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct PutCommitRequest {
    pub key: ObjectKey,
    pub put_id: PutId,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct GetRouteResponse {
    pub key: ObjectKey,
    pub worker_id: WorkerId,
    pub worker_addr: String,
    pub object_generation: u64,
    pub lease_id: LeaseId,
    pub lease_expires_ms: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Round-trips a value through JSON and asserts equality.
    fn round_trip<T>(value: &T)
    where
        T: serde::Serialize + serde::de::DeserializeOwned + PartialEq + std::fmt::Debug,
    {
        let json = serde_json::to_string(value).expect("serialize");
        let back: T = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(*value, back);
    }

    #[test]
    fn object_state_round_trips() {
        for state in [
            ObjectState::Writing,
            ObjectState::Complete,
            ObjectState::Failed,
            ObjectState::Removed,
        ] {
            round_trip(&state);
        }
    }

    #[test]
    fn placement_round_trips() {
        round_trip(&Placement {
            worker_id: "w1".to_string(),
            worker_epoch: 3,
            object_generation: 7,
        });
    }

    #[test]
    fn register_request_round_trips() {
        round_trip(&RegisterRequest {
            worker_id: "w1".to_string(),
            addr: "http://127.0.0.1:9000".to_string(),
            capacity_bytes: 1024,
        });
    }

    #[test]
    fn register_response_round_trips() {
        round_trip(&RegisterResponse { epoch: 1 });
    }

    #[test]
    fn heartbeat_request_round_trips() {
        round_trip(&HeartbeatRequest {
            worker_id: "w1".to_string(),
            epoch: 1,
        });
    }

    #[test]
    fn put_start_request_round_trips() {
        round_trip(&PutStartRequest {
            key: "kv://v1/model/m/prefix/abc/tokens/64".to_string(),
            size_bytes: 4096,
        });
    }

    #[test]
    fn put_start_response_round_trips() {
        round_trip(&PutStartResponse {
            put_id: uuid::Uuid::new_v4(),
            worker_id: "w1".to_string(),
            worker_addr: "http://127.0.0.1:9000".to_string(),
            object_generation: 2,
        });
    }

    #[test]
    fn put_commit_request_round_trips() {
        round_trip(&PutCommitRequest {
            key: "k".to_string(),
            put_id: uuid::Uuid::new_v4(),
        });
    }

    #[test]
    fn get_route_response_round_trips() {
        round_trip(&GetRouteResponse {
            key: "k".to_string(),
            worker_id: "w1".to_string(),
            worker_addr: "http://127.0.0.1:9000".to_string(),
            object_generation: 5,
            lease_id: uuid::Uuid::new_v4(),
            lease_expires_ms: 1_700_000_000_000,
        });
    }
}
