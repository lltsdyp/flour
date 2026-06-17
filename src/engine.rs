use std::path::Path;

use anyhow::Context;
use candle_core::{DType, Device, IndexOp, Tensor};

use crate::loader::safetensors::load_var_builder;
use crate::models::{self, common::{CausalLM, Cache, EosTokenId}};
use crate::sampling::{apply_repeat_penalty, LogitsSampler, SamplingParams};
use crate::tokenizer::{ChatMessage, ChatTemplate, Tokenizer};

pub struct GenerationStats {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
}

pub struct Engine {
    model: CausalLM,
    tokenizer: Tokenizer,
    chat_template: ChatTemplate,
    eos_token_id: EosTokenId,
    device: Device,
    model_id: String,
}

impl Engine {
    pub fn load(model_dir: &Path) -> anyhow::Result<Self> {
        let config_path = model_dir.join("config.json");
        let raw: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(&config_path)
                .with_context(|| format!("reading {}", config_path.display()))?,
        )?;
        let family = models::detect_family(&raw)?;
        let cfg = models::load_config(family, &raw)?;

        let device = Device::Cpu;
        let vb = load_var_builder(model_dir, DType::F32, &device)?;
        let model = CausalLM::load(vb, cfg.clone())?;

        let tokenizer = Tokenizer::from_file(&model_dir.join("tokenizer.json"))?;
        let chat_template = ChatTemplate::for_family(family);

        let eos_token_id = cfg.eos_token_id.clone().unwrap_or_else(|| {
            let fallback = match chat_template {
                ChatTemplate::Llama3 => "<|eot_id|>",
                ChatTemplate::ChatMl => "<|im_end|>",
            };
            EosTokenId::Single(tokenizer.token_to_id(fallback).unwrap_or(0))
        });

        let model_id = model_dir
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "model".to_string());

        Ok(Self { model, tokenizer, chat_template, eos_token_id, device, model_id })
    }

    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    pub fn generate(
        &self,
        messages: &[ChatMessage],
        params: &SamplingParams,
        mut on_token: impl FnMut(&str),
    ) -> anyhow::Result<GenerationStats> {
        let prompt = self.chat_template.render(messages);
        let prompt_tokens = self.tokenizer.encode(&prompt)?;
        let prompt_len = prompt_tokens.len();

        let mut cache = Cache::new(self.model.config(), &self.device)?;
        let mut sampler = LogitsSampler::new(params.seed);
        let mut all_tokens = prompt_tokens.clone();

        let input = Tensor::new(prompt_tokens.as_slice(), &self.device)?.unsqueeze(0)?;
        let mut logits = self.model.forward(&input, 0, &mut cache)?;
        let mut index_pos = prompt_len;
        let mut completion_tokens = 0usize;

        for _ in 0..params.max_tokens {
            let seq_len = logits.dim(1)?;
            let last = logits.i((0, seq_len - 1))?.to_dtype(DType::F32)?;
            let mut logits_vec = last.to_vec1::<f32>()?;

            if params.repeat_penalty != 1.0 {
                let start = all_tokens.len().saturating_sub(params.repeat_last_n);
                apply_repeat_penalty(&mut logits_vec, params.repeat_penalty, &all_tokens[start..]);
            }

            let next_token = sampler.sample(&logits_vec, params);
            if self.eos_token_id.is_eos(next_token) {
                break;
            }

            let piece = self.tokenizer.decode(&[next_token])?;
            on_token(&piece);
            all_tokens.push(next_token);
            completion_tokens += 1;

            let next_input = Tensor::new(&[next_token], &self.device)?.unsqueeze(0)?;
            logits = self.model.forward(&next_input, index_pos, &mut cache)?;
            index_pos += 1;
        }

        Ok(GenerationStats { prompt_tokens: prompt_len, completion_tokens })
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::tokenizer::ChatMessage;
    use candle_core::{DType, Device, Tensor};
    use std::collections::HashMap;

    pub(crate) fn fixture_dir_for_external_use() -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        write_fixture_model(dir.path());
        dir
    }

    /// Builds a tiny but structurally valid Qwen3-family model directory: config.json,
    /// a minimal tokenizer.json (byte-level BPE over a handful of fixed tokens), and
    /// randomly-initialized safetensors weights matching that config.
    fn write_fixture_model(dir: &std::path::Path) {
        let hidden = 8usize;
        let heads = 2usize;
        let kv_heads = 1usize;
        let head_dim = 4usize;
        let intermediate = 16usize;
        let layers = 1usize;
        // 128 = full ASCII byte range, so every byte that can appear in a rendered ChatML
        // prompt (letters, digits, punctuation, "<|...|>", newlines) has a vocab entry.
        let vocab = 128usize;

        let config = serde_json::json!({
            "architectures": ["Qwen3ForCausalLM"],
            "hidden_size": hidden,
            "intermediate_size": intermediate,
            "vocab_size": vocab,
            "num_hidden_layers": layers,
            "num_attention_heads": heads,
            "num_key_value_heads": kv_heads,
            "head_dim": head_dim,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0,
            "tie_word_embeddings": true,
            "max_position_embeddings": 64
        });
        std::fs::write(dir.join("config.json"), serde_json::to_string(&config).unwrap()).unwrap();

        fn rand_tensor(shape: &[usize], seed: u64) -> Tensor {
            use rand::rngs::StdRng;
            use rand::{Rng, SeedableRng};
            let numel: usize = shape.iter().product();
            let mut rng = StdRng::seed_from_u64(seed);
            let data: Vec<f32> = (0..numel).map(|_| rng.gen_range(-0.05..0.05)).collect();
            Tensor::from_vec(data, shape, &Device::Cpu).unwrap()
        }

        let size_q = head_dim * heads;
        let size_kv = head_dim * kv_heads;
        let mut tensors: HashMap<String, Tensor> = HashMap::new();
        tensors.insert("model.embed_tokens.weight".into(), rand_tensor(&[vocab, hidden], 1));
        tensors.insert("model.norm.weight".into(), Tensor::ones(hidden, DType::F32, &Device::Cpu).unwrap());
        for l in 0..layers {
            let p = format!("model.layers.{l}");
            tensors.insert(format!("{p}.input_layernorm.weight"), Tensor::ones(hidden, DType::F32, &Device::Cpu).unwrap());
            tensors.insert(format!("{p}.post_attention_layernorm.weight"), Tensor::ones(hidden, DType::F32, &Device::Cpu).unwrap());
            tensors.insert(format!("{p}.self_attn.q_proj.weight"), rand_tensor(&[size_q, hidden], 10));
            tensors.insert(format!("{p}.self_attn.k_proj.weight"), rand_tensor(&[size_kv, hidden], 11));
            tensors.insert(format!("{p}.self_attn.v_proj.weight"), rand_tensor(&[size_kv, hidden], 12));
            tensors.insert(format!("{p}.self_attn.o_proj.weight"), rand_tensor(&[hidden, size_q], 13));
            tensors.insert(format!("{p}.self_attn.q_norm.weight"), Tensor::ones(head_dim, DType::F32, &Device::Cpu).unwrap());
            tensors.insert(format!("{p}.self_attn.k_norm.weight"), Tensor::ones(head_dim, DType::F32, &Device::Cpu).unwrap());
            tensors.insert(format!("{p}.mlp.gate_proj.weight"), rand_tensor(&[intermediate, hidden], 14));
            tensors.insert(format!("{p}.mlp.up_proj.weight"), rand_tensor(&[intermediate, hidden], 15));
            tensors.insert(format!("{p}.mlp.down_proj.weight"), rand_tensor(&[hidden, intermediate], 16));
        }
        candle_core::safetensors::save(&tensors, dir.join("model.safetensors")).unwrap();

        // Minimal byte-level tokenizer: vocab of single bytes 0..vocab, no merges, so any
        // input string encodes to one token per byte and round-trips exactly. The `tokenizers`
        // crate's ByteLevel pre-tokenizer remaps raw bytes through the GPT-2 byte-to-unicode
        // table before lookup (printable bytes map to themselves; everything else, including
        // all of 0..=32, maps into the U+0100+ range) — so the vocab keys must go through the
        // same mapping, not the raw byte values, or encoding silently produces zero tokens.
        fn bytes_to_unicode() -> HashMap<u8, char> {
            let mut bs: Vec<u8> = vec![];
            bs.extend(b'!'..=b'~');
            bs.extend(0xA1u8..=0xACu8);
            bs.extend(0xAEu8..=0xFFu8);
            let mut cs: Vec<u32> = bs.iter().map(|&b| b as u32).collect();
            let mut n = 0u32;
            for b in 0u8..=255 {
                if !bs.contains(&b) {
                    bs.push(b);
                    cs.push(256 + n);
                    n += 1;
                }
            }
            bs.into_iter().zip(cs).map(|(b, c)| (b, char::from_u32(c).unwrap())).collect()
        }
        let byte_map = bytes_to_unicode();
        let mut vocab_map = serde_json::Map::new();
        for i in 0..vocab {
            let c = byte_map[&(i as u8)];
            vocab_map.insert(c.to_string(), serde_json::json!(i));
        }
        let tokenizer_json = serde_json::json!({
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [],
            "normalizer": null,
            "pre_tokenizer": { "type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": false },
            "post_processor": null,
            "decoder": { "type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": false },
            "model": {
                "type": "BPE",
                "dropout": null,
                "unk_token": null,
                "continuing_subword_prefix": null,
                "end_of_word_suffix": null,
                "fuse_unk": false,
                "byte_fallback": false,
                "vocab": vocab_map,
                "merges": []
            }
        });
        std::fs::write(dir.join("tokenizer.json"), serde_json::to_string(&tokenizer_json).unwrap()).unwrap();
    }

    #[test]
    fn load_and_generate_end_to_end_with_tiny_random_model() {
        let dir = tempfile::tempdir().unwrap();
        write_fixture_model(dir.path());

        let engine = Engine::load(dir.path()).unwrap();
        assert_eq!(engine.model_id(), dir.path().file_name().unwrap().to_string_lossy());

        let messages = vec![ChatMessage { role: "user".into(), content: "hi".into() }];
        let params = crate::sampling::SamplingParams { max_tokens: 4, seed: 1, ..Default::default() };

        let mut produced = String::new();
        let stats = engine.generate(&messages, &params, |tok| produced.push_str(tok)).unwrap();

        assert!(stats.prompt_tokens > 0);
        assert!(stats.completion_tokens <= 4);
    }

    #[test]
    fn generate_is_deterministic_for_a_fixed_seed() {
        let dir = tempfile::tempdir().unwrap();
        write_fixture_model(dir.path());
        let engine = Engine::load(dir.path()).unwrap();
        let messages = vec![ChatMessage { role: "user".into(), content: "hi".into() }];
        let params = crate::sampling::SamplingParams { max_tokens: 4, seed: 99, temperature: 0.0, ..Default::default() };

        let mut a = String::new();
        engine.generate(&messages, &params, |tok| a.push_str(tok)).unwrap();
        let mut b = String::new();
        engine.generate(&messages, &params, |tok| b.push_str(tok)).unwrap();
        assert_eq!(a, b);
    }
}
