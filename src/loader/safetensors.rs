use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context};
use candle_core::{DType, Device};
use candle_nn::VarBuilder;

pub fn load_var_builder(
    model_dir: &Path,
    dtype: DType,
    device: &Device,
) -> anyhow::Result<VarBuilder<'static>> {
    let files = collect_safetensors_files(model_dir)?;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&files, dtype, device) }
        .context("failed to mmap safetensors weights")?;
    Ok(vb)
}

fn collect_safetensors_files(model_dir: &Path) -> anyhow::Result<Vec<PathBuf>> {
    let index_path = model_dir.join("model.safetensors.index.json");
    if index_path.exists() {
        let raw = std::fs::read_to_string(&index_path)
            .with_context(|| format!("reading {}", index_path.display()))?;
        let json: serde_json::Value = serde_json::from_str(&raw)?;
        let weight_map = json
            .get("weight_map")
            .and_then(|v| v.as_object())
            .ok_or_else(|| anyhow!("{} missing 'weight_map'", index_path.display()))?;

        let mut files: Vec<String> = weight_map
            .values()
            .filter_map(|v| v.as_str().map(String::from))
            .collect();
        files.sort();
        files.dedup();
        Ok(files.into_iter().map(|f| model_dir.join(f)).collect())
    } else {
        let single = model_dir.join("model.safetensors");
        if !single.exists() {
            return Err(anyhow!(
                "no model.safetensors or model.safetensors.index.json found in {}",
                model_dir.display()
            ));
        }
        Ok(vec![single])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};
    use std::collections::HashMap;

    #[test]
    fn loads_single_safetensors_file() {
        let dir = tempfile::tempdir().unwrap();
        let mut tensors = HashMap::new();
        tensors.insert(
            "weight".to_string(),
            Tensor::ones((2, 2), DType::F32, &Device::Cpu).unwrap(),
        );
        candle_core::safetensors::save(&tensors, dir.path().join("model.safetensors")).unwrap();

        let vb = load_var_builder(dir.path(), DType::F32, &Device::Cpu).unwrap();
        let t = vb.get((2, 2), "weight").unwrap();
        assert_eq!(t.dims(), &[2, 2]);
    }

    #[test]
    fn loads_sharded_safetensors_via_index_json() {
        let dir = tempfile::tempdir().unwrap();

        let mut shard0 = HashMap::new();
        shard0.insert(
            "a".to_string(),
            Tensor::ones((1,), DType::F32, &Device::Cpu).unwrap(),
        );
        candle_core::safetensors::save(
            &shard0,
            dir.path().join("model-00001-of-00002.safetensors"),
        )
        .unwrap();

        let mut shard1 = HashMap::new();
        shard1.insert(
            "b".to_string(),
            Tensor::zeros((1,), DType::F32, &Device::Cpu).unwrap(),
        );
        candle_core::safetensors::save(
            &shard1,
            dir.path().join("model-00002-of-00002.safetensors"),
        )
        .unwrap();

        let index = serde_json::json!({
            "weight_map": {
                "a": "model-00001-of-00002.safetensors",
                "b": "model-00002-of-00002.safetensors",
            }
        });
        std::fs::write(
            dir.path().join("model.safetensors.index.json"),
            serde_json::to_string(&index).unwrap(),
        )
        .unwrap();

        let vb = load_var_builder(dir.path(), DType::F32, &Device::Cpu).unwrap();
        assert!(vb.get((1,), "a").is_ok());
        assert!(vb.get((1,), "b").is_ok());
    }

    #[test]
    fn errors_when_no_weights_present() {
        let dir = tempfile::tempdir().unwrap();
        assert!(load_var_builder(dir.path(), DType::F32, &Device::Cpu).is_err());
    }
}
