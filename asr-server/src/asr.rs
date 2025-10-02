// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

use crate::AsrStreamingQuery as Query;
use anyhow::{bail, Context, Result};
use axum::extract::ws;
use candle::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::quantized_var_builder::VarBuilder as QuantizedVarBuilder;
use std::collections::VecDeque;
use tokio::time::{timeout, Duration};

const FRAME_SIZE: usize = 1920;

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(tag = "type")]
pub enum InMsg {
    Init,
    Marker { id: i64 },
    Audio { pcm: Vec<f32> },
    OggOpus { data: Vec<u8> },
}

impl InMsg {
    /// Decode a websocket binary payload into an `InMsg` using the requested encoding.
    pub fn from_ws_payload(payload: &[u8], expect_pcm16le: bool) -> Result<Self> {
        if expect_pcm16le {
            return Self::from_pcm16le(payload);
        }
        Ok(rmp_serde::from_slice(payload)?)
    }

    fn from_pcm16le(payload: &[u8]) -> Result<Self> {
        if payload.len() % 2 != 0 {
            bail!("received PCM16LE payload with odd length {}", payload.len());
        }
        let mut pcm = Vec::with_capacity(payload.len() / 2);
        for chunk in payload.chunks_exact(2) {
            let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
            pcm.push(sample as f32 / 32768.0);
        }
        Ok(Self::Audio { pcm })
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type")]
pub enum OutMsg {
    Word { text: String, start_time: f64 },
    EndWord { stop_time: f64 },
    Marker { id: i64 },
    Step { step_idx: usize, prs: Vec<f32>, buffered_pcm: usize },
    Error { message: String },
    Ready,
}

#[derive(Debug)]
pub struct Asr {
    asr_delay_in_tokens: usize,
    temperature: f64,
    lm: asr_core::lm::LmModel,
    audio_tokenizer: asr_core::mimi::Mimi,
    text_tokenizer: std::sync::Arc<sentencepiece::SentencePieceProcessor>,
    instance_name: String,
    log_dir: std::path::PathBuf,
    conditions: Option<asr_core::conditioner::Condition>,
}

impl Asr {
    pub fn new(asr: &crate::AsrConfig, config: &crate::Config, dev: &Device) -> Result<Self> {
        let lm_vb = if std::path::Path::new(&asr.lm_model_file)
            .extension()
            .is_some_and(|ext| ext == "gguf")
        {
            let vb = QuantizedVarBuilder::from_gguf(&asr.lm_model_file, dev)?;
            asr_core::nn::MaybeQuantizedVarBuilder::Quantized(vb)
        } else {
            let dtype = crate::utils::model_dtype(asr.dtype_override.as_deref(), dev)?;
            let vb =
                unsafe { VarBuilder::from_mmaped_safetensors(&[&asr.lm_model_file], dtype, dev)? };
            asr_core::nn::MaybeQuantizedVarBuilder::Real(vb)
        };
        let lm = asr_core::lm::LmModel::new(&asr.model, lm_vb)?;
        let conditions = match lm.condition_provider() {
            None => None,
            Some(cp) => {
                let delay =
                    asr.conditioning_delay.context("missing conditioning_delay in config")?;
                let conditions = cp.condition_cont("delay", -delay)?;
                tracing::info!(?conditions, "generated conditions");
                Some(conditions)
            }
        };
        let audio_tokenizer = {
            let vb = unsafe {
                candle_nn::VarBuilder::from_mmaped_safetensors(
                    &[&asr.audio_tokenizer_file],
                    DType::F32,
                    dev,
                )?
            };
            let mut cfg = asr_core::mimi::Config::v0_1(Some(asr.model.audio_codebooks));
            // The mimi transformer runs at 25Hz.
            cfg.transformer.max_seq_len = asr.model.transformer.max_seq_len * 2;
            asr_core::mimi::Mimi::new(cfg, vb)?
        };
        let text_tokenizer = sentencepiece::SentencePieceProcessor::open(&asr.text_tokenizer_file)
            .with_context(|| asr.text_tokenizer_file.clone())?;
        Ok(Self {
            asr_delay_in_tokens: asr.asr_delay_in_tokens,
            lm,
            temperature: asr.temperature.unwrap_or(0.0),
            audio_tokenizer,
            text_tokenizer: text_tokenizer.into(),
            log_dir: config.log_dir.clone().into(),
            instance_name: config.instance_name.clone(),
            conditions,
        })
    }

    pub fn warmup(&self) -> Result<()> {
        let lm = self.lm.clone();
        let audio_tokenizer = self.audio_tokenizer.clone();
        let mut state = asr_core::asr::State::new(
            1,
            self.asr_delay_in_tokens,
            self.temperature,
            audio_tokenizer,
            lm,
        )?;
        let dev = state.device().clone();
        let pcm = vec![0f32; FRAME_SIZE * state.batch_size()];
        for _ in 0..2 {
            let pcm = Tensor::new(pcm.as_slice(), &dev)?.reshape((state.batch_size(), 1, ()))?;
            let _asr_msgs =
                state.step_pcm(pcm, self.conditions.as_ref(), &().into(), |_, _, _| ())?;
        }
        Ok(())
    }

    pub async fn handle_socket(&self, socket: ws::WebSocket, query: Query) -> Result<()> {
        use futures_util::{SinkExt, StreamExt};
        use serde::Serialize;

        let (mut sender, mut receiver) = socket.split();
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<OutMsg>();
        let (log_tx, log_rx) = std::sync::mpsc::channel();
        let lm = self.lm.clone();
        let audio_tokenizer = self.audio_tokenizer.clone();
        let mut state = asr_core::asr::State::new(
            1,
            self.asr_delay_in_tokens,
            self.temperature,
            audio_tokenizer,
            lm,
        )?;
        let text_tokenizer = self.text_tokenizer.clone();

        let asr_delay_in_tokens = self.asr_delay_in_tokens;
        let conditions = self.conditions.clone();
        let mut ogg_opus_decoder = kaudio::ogg_opus::Decoder::new(24000, 1920)?;
        let expect_pcm16le = true;
        let disable_msgpack = true;
        let disable_step = true;
        let recv_loop = crate::utils::spawn("recv_loop", async move {
            let dev = state.device().clone();
            // Store the markers in a double ended queue
            let mut markers = VecDeque::new();
            while let Some(msg) = receiver.next().await {
                let msg = match msg? {
                    ws::Message::Binary(x) => x,
                    // ping messages are automatically answered by tokio-tungstenite as long as
                    // the connection is read from.
                    ws::Message::Ping(_) | ws::Message::Pong(_) | ws::Message::Text(_) => continue,
                    ws::Message::Close(_) => break,
                };
                let msg = InMsg::from_ws_payload(&msg, expect_pcm16le)?;
                let pcm = match msg {
                    // Init is only used in batched mode.
                    InMsg::Init => None,
                    InMsg::Marker { id } => {
                        tracing::info!("received marker {id}");
                        let step_idx = state.model_step_idx();
                        markers.push_back((step_idx, id));
                        None
                    }
                    InMsg::OggOpus { data } => ogg_opus_decoder.decode(&data)?.map(|v| v.to_vec()),
                    InMsg::Audio { pcm } => Some(pcm),
                };
                if let Some(pcm) = pcm {
                    tracing::info!("received audio {}", pcm.len());
                    let pcm = Tensor::new(pcm.as_slice(), &dev)?
                        .reshape((1, 1, ()))?
                        .broadcast_as((state.batch_size(), 1, pcm.len()))?;
                    let asr_msgs = state.step_pcm(
                        pcm,
                        conditions.as_ref(),
                        &().into(),
                        |_, text_tokens, audio_tokens| {
                            let res = || {
                                let text_tokens = text_tokens.to_device(&Device::Cpu)?;
                                let audio_tokens: Vec<Tensor> = audio_tokens
                                    .iter()
                                    .map(|t| t.to_device(&Device::Cpu))
                                    .collect::<candle::Result<Vec<_>>>()?;
                                let audio_tokens = Tensor::stack(&audio_tokens, 1)?;
                                log_tx.send((text_tokens, audio_tokens))?;
                                Ok::<_, anyhow::Error>(())
                            };
                            if let Err(err) = res() {
                                tracing::error!(?err, "failed to send log");
                            }
                        },
                    )?;
                    for asr_msg in asr_msgs.into_iter() {
                        let msg = match asr_msg {
                            asr_core::asr::AsrMsg::Word { tokens, start_time, .. } => {
                                OutMsg::Word {
                                    text: text_tokenizer.decode_piece_ids(&tokens)?,
                                    start_time,
                                }
                            }
                            asr_core::asr::AsrMsg::Step { step_idx, prs } => {
                                if disable_step {
                                    continue;
                                }
                                let prs = prs.iter().map(|p| p[0]).collect::<Vec<_>>();
                                OutMsg::Step { step_idx, prs, buffered_pcm: 0 }
                            }
                            asr_core::asr::AsrMsg::EndWord { stop_time, .. } => {
                                OutMsg::EndWord { stop_time }
                            }
                        };
                        tx.send(msg)?
                    }
                    while let Some((step_idx, id)) = markers.front() {
                        if *step_idx + asr_delay_in_tokens <= state.model_step_idx() {
                            tx.send(OutMsg::Marker { id: *id })?;
                            markers.pop_front();
                        } else {
                            break;
                        }
                    }
                }
            }
            Ok::<(), anyhow::Error>(())
        });
        let send_loop = crate::utils::spawn("send_loop", async move {
            loop {
                // The recv method is cancel-safe so can be wrapped in a timeout.
                let msg = timeout(Duration::from_secs(10), rx.recv()).await;
                let msg = match msg {
                    Ok(None) => break,
                    Err(_) => Some(ws::Message::Ping(vec![].into())),
                    Ok(Some(msg)) => {
                        if disable_step && matches!(msg, OutMsg::Step { .. }) {
                            continue;
                        }
                        if disable_msgpack && matches!(msg, OutMsg::Ready) {
                            continue;
                        }
                        if disable_msgpack && matches!(msg, OutMsg::EndWord { .. }) {
                            continue;
                        }
                        if disable_msgpack {
                            let payload = serde_json::to_string(&msg)?;
                            sender.send(ws::Message::Text(payload.into())).await?;
                            continue;
                        }
                        let mut buf = vec![];
                        msg.serialize(
                            &mut rmp_serde::Serializer::new(&mut buf)
                                .with_human_readable()
                                .with_struct_map(),
                        )?;
                        Some(ws::Message::Binary(buf.into()))
                    }
                };
                if let Some(msg) = msg {
                    sender.send(msg).await?;
                }
            }
            tracing::info!("send loop exited");
            Ok::<(), anyhow::Error>(())
        });
        let sleep = tokio::time::sleep(std::time::Duration::from_secs(360));
        tokio::pin!(sleep);
        // select should ensure that all the threads get aborted on timeout.
        // TODO(laurent): this actually doesn't work as expected, and the background threads don't
        // appear to be cancelled properly (at least the websocket connection remains open.
        // laurent: Actually I guess this is because we wait for at least one of these to finish
        // before exiting this task.
        tokio::select! {
            _ = &mut sleep => {
                tracing::error!("reached timeout");
            }
            _ = recv_loop => {
            }
            _ = send_loop => {
            }
        }
        let (text_tokens, audio_tokens): (Vec<_>, Vec<_>) = log_rx.try_iter().unzip();
        let text_tokens = Tensor::cat(&text_tokens, candle::D::Minus1)?;
        let audio_tokens = Tensor::cat(&audio_tokens, candle::D::Minus1)?;
        self.save_logs(&query, audio_tokens, text_tokens)?;
        tracing::info!("exiting handle-socket");
        Ok(())
    }

    fn save_logs(&self, query: &Query, audio_tokens: Tensor, text_tokens: Tensor) -> Result<()> {
        let since_epoch = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?;
        let (secs, us) = (since_epoch.as_secs(), since_epoch.subsec_micros());
        let base_path = self.log_dir.join(format!("{}-asr-{secs}-{us}", self.instance_name));
        let json_filename = base_path.with_extension("json");
        let json_content = serde_json::to_string_pretty(query)?;
        std::fs::write(json_filename, json_content)?;
        let st_filename = base_path.with_extension("safetensors");
        let audio_tokens = audio_tokens.to_device(&Device::Cpu)?.to_dtype(DType::I64)?;
        let st_content =
            std::collections::HashMap::from([("text", text_tokens), ("audio", audio_tokens)]);
        candle::safetensors::save(&st_content, st_filename)?;
        Ok(())
    }
}
