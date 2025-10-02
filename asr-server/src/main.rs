// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

use anyhow::Result;
use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
};
use candle::Device;
use std::str::FromStr;
use std::sync::Arc;

mod asr;
mod batched_asr;
mod metrics;
mod utils;

#[derive(clap::Parser, Debug)]
struct WorkerArgs {
    #[clap(short = 'l', long = "log", default_value = "info")]
    log_level: String,

    #[clap(short = 'a', long = "addr", default_value = "0.0.0.0")]
    addr: String,

    #[clap(short = 'p', long = "port", default_value = "8080")]
    port: u16,

    #[clap(long)]
    cpu: bool,

    #[clap(long)]
    config: String,

    #[clap(long)]
    silent: bool,
}

#[derive(Debug, clap::Subcommand)]
enum Command {
    Validate { configs: Vec<String> },
    Worker(WorkerArgs),
}

#[derive(clap::Parser, Debug)]
#[clap(name = "server", about = "Kyutai asr-core server")]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct AsrConfig {
    pub lm_model_file: String,
    pub text_tokenizer_file: String,
    pub audio_tokenizer_file: String,
    pub model: asr_core::lm::Config,
    pub asr_delay_in_tokens: usize,
    #[serde(default)]
    pub log_frequency_s: Option<f64>,
    #[serde(default)]
    pub conditioning_delay: Option<f32>,
    // The default for bools in rust is false.
    #[serde(default)]
    pub conditioning_learnt_padding: bool,
    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub dtype_override: Option<String>,
}

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(tag = "type")]
pub enum ModuleConfig {
    Asr {
        path: String,
        #[serde(flatten)]
        config: AsrConfig,
    },
    BatchedAsr {
        path: String,
        #[serde(flatten)]
        config: AsrConfig,
        batch_size: usize,
    },
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct Config {
    pub static_dir: String,
    pub log_dir: String,
    pub instance_name: String,
    #[serde(default)]
    pub modules: std::collections::HashMap<String, ModuleConfig>,
    pub authorized_ids: std::collections::HashSet<String>,
}

impl Config {
    pub fn load<P: AsRef<std::path::Path>>(p: P) -> Result<Self> {
        use utils::resolve_or_download as rod;
        let config = std::fs::read_to_string(p)?;
        let mut config: Self = toml::from_str(&config)?;
        for (_, c) in config.modules.iter_mut() {
            match c {
                ModuleConfig::BatchedAsr { path: _, config: c, batch_size: _ } => {
                    c.lm_model_file = rod(&c.lm_model_file)?;
                    c.text_tokenizer_file = rod(&c.text_tokenizer_file)?;
                    c.audio_tokenizer_file = rod(&c.audio_tokenizer_file)?;
                }
                ModuleConfig::Asr { path: _, config: c } => {
                    c.lm_model_file = rod(&c.lm_model_file)?;
                    c.text_tokenizer_file = rod(&c.text_tokenizer_file)?;
                    c.audio_tokenizer_file = rod(&c.audio_tokenizer_file)?;
                }
            }
        }
        config.static_dir = rod(&config.static_dir)?;
        config.log_dir = rod(&config.log_dir)?;
        config.instance_name = rod(&config.instance_name)?;
        Ok(config)
    }
}

fn device(cpu: bool) -> Result<Device> {
    if cpu {
        Ok(Device::Cpu)
    } else if candle::utils::cuda_is_available() {
        Ok(Device::new_cuda(0)?)
    } else if candle::utils::metal_is_available() {
        Ok(Device::new_metal(0)?)
    } else {
        Ok(Device::Cpu)
    }
}

enum Module {
    Asr { path: String, m: Arc<asr::Asr> },
    BatchedAsr { path: String, m: Arc<batched_asr::BatchedAsr> },
}

struct SharedStateInner {
    _config: Config,
}

type SharedState = Arc<SharedStateInner>;

impl Module {
    fn new(module_cfg: &ModuleConfig, full_cfg: &Config, dev: &Device) -> Result<Self> {
        let m = match module_cfg {
            ModuleConfig::Asr { path, config } => {
                let m = asr::Asr::new(config, full_cfg, dev)?;
                let m = Arc::new(m);
                tracing::info!("warming up the asr");
                m.warmup()?;
                tracing::info!("done warming up the asr, ready to roll!");
                Self::Asr { m, path: path.to_string() }
            }
            ModuleConfig::BatchedAsr { path, config, batch_size } => {
                let m = batched_asr::BatchedAsr::new(*batch_size, config, full_cfg, dev)?;
                let m = Arc::new(m);
                Self::BatchedAsr { m, path: path.to_string() }
            }
        };
        Ok(m)
    }

    fn router(&self, shared_state: &SharedState) -> Result<axum::Router<()>> {
        let router = match self {
            Self::Asr { path, m } => asr_router(m.clone(), path, shared_state),
            Self::BatchedAsr { path, m } => batched_asr_router(m.clone(), path, shared_state),
        };
        Ok(router)
    }
}

struct AppStateInner {
    modules: Vec<Module>,
}

type AppState = Arc<AppStateInner>;

impl AppStateInner {
    fn new(args: &WorkerArgs, config: Config) -> Result<Self> {
        let device = device(args.cpu)?;

        // The following does not have a significant impact as soon as batch sizes are
        // large enough so we don't activate it for now.
        // #[cfg(feature = "cuda")]
        // if let candle::Device::Cuda(d) = &device {
        //     unsafe {
        //         d.disable_event_tracking();
        //     }
        // };

        let mut modules = Vec::with_capacity(config.modules.len());
        for (_, module_cfg) in config.modules.iter() {
            let m = Module::new(module_cfg, &config, &device)?;
            modules.push(m)
        }
        Ok(Self { modules })
    }
}

fn tracing_init(
    log_dir: &str,
    instance_name: &str,
    log_level: &str,
    silent: bool,
) -> Result<tracing_appender::non_blocking::WorkerGuard> {
    use tracing_subscriber::prelude::*;

    let build_info = utils::BuildInfo::new();
    let file_appender = tracing_appender::rolling::daily(log_dir, format!("log.{instance_name}"));
    let (non_blocking, guard) = tracing_appender::non_blocking(file_appender);
    let filter = tracing_subscriber::filter::LevelFilter::from_str(log_level)?;
    let mut layers = vec![tracing_subscriber::fmt::layer()
        .event_format(tracing_subscriber::fmt::format().with_file(true).with_line_number(true))
        .with_writer(non_blocking)
        .with_filter(filter)
        .boxed()];
    if !silent {
        layers.push(Box::new(
            tracing_subscriber::fmt::layer()
                .event_format(
                    tracing_subscriber::fmt::format().with_file(true).with_line_number(true),
                )
                .with_writer(std::io::stdout)
                .with_filter(filter),
        ))
    };
    tracing_subscriber::registry().with(layers).init();
    tracing::info!(?build_info);
    Ok(guard)
}

async fn metrics(
    axum::extract::ConnectInfo(_addr): axum::extract::ConnectInfo<std::net::SocketAddr>,
    _state: axum::extract::State<AppState>,
    _req: axum::extract::Query<()>,
) -> impl IntoResponse {
    use prometheus::Encoder;

    let encoder = prometheus::TextEncoder::new();
    let metric_families = prometheus::gather();
    let mut buffer = vec![];
    if let Err(err) = encoder.encode(&metric_families, &mut buffer) {
        return (axum::http::StatusCode::INTERNAL_SERVER_ERROR, err.to_string()).into_response();
    };
    axum::response::Response::builder()
        .status(200)
        .header(axum::http::header::CONTENT_TYPE, encoder.format_type())
        .body(axum::body::Body::from(buffer))
        .unwrap()
}

#[tokio::main(flavor = "multi_thread")]
async fn main() {
    // When an error bubbles up in the tokio main function, the whole program does not
    // seem to crash if some background tasks are still running.
    // This can lead to errors such as "port already in use" not being reported so we
    // exit the process explicitely here.
    if let Err(err) = main_().await {
        eprintln!("Error: {err}");
        std::process::exit(1);
    }
}

async fn main_() -> Result<()> {
    let args = <Args as clap::Parser>::parse();
    match args.command {
        Command::Validate { configs } => {
            tracing_subscriber::fmt().init();
            for config in configs.iter() {
                let _ = Config::load(config)?;
                tracing::info!(?config, "loaded succesfully")
            }
        }
        Command::Worker(args) => {
            use axum::routing::get;

            let config = Config::load(&args.config)?;
            if std::env::var("RUST_LOG").is_err() {
                std::env::set_var("RUST_LOG", format!("{},hyper=info,mio=info", args.log_level))
            }
            let _guard =
                tracing_init(&config.log_dir, &config.instance_name, &args.log_level, args.silent)?;
            let num_workers = tokio::runtime::Handle::current().metrics().num_workers();
            tracing::info!(num_workers, "starting worker");

            let static_dir = utils::resolve_or_download(&config.static_dir)?;
            let shared_state = Arc::new(SharedStateInner { _config: config.clone() });
            let state = Arc::new(AppStateInner::new(&args, config)?);
            let mut app = axum::Router::new()
                .route("/api/build_info", get(build_info))
                .route("/api/modules_info", get(modules_info))
                .route("/metrics", axum::routing::get(metrics))
                .fallback_service(
                    tower_http::services::ServeDir::new(&static_dir)
                        .append_index_html_on_directories(true),
                )
                .layer(
                    tower::ServiceBuilder::new()
                        .layer(tower_http::trace::TraceLayer::new_for_http()),
                )
                .with_state(state.clone());
            for module in state.modules.iter() {
                app = app.merge(module.router(&shared_state)?)
            }

            let sock_addr = std::net::SocketAddr::from((
                std::net::IpAddr::from_str(args.addr.as_str())
                    .unwrap_or(std::net::IpAddr::V6(std::net::Ipv6Addr::LOCALHOST)),
                args.port,
            ));
            tracing::info!("listening on http://{}", sock_addr);
            let listener = tokio::net::TcpListener::bind(sock_addr).await?;
            axum::serve(
                listener,
                app.into_make_service_with_connect_info::<std::net::SocketAddr>(),
            )
            .await?;
        }
    }
    Ok(())
}

async fn build_info(
    axum::extract::ConnectInfo(_addr): axum::extract::ConnectInfo<std::net::SocketAddr>,
    _state: axum::extract::State<AppState>,
    _req: axum::extract::Query<()>,
) -> impl IntoResponse {
    let build_info = utils::BuildInfo::new();
    utils::WrapJson(Ok(build_info)).into_response()
}

async fn modules_info(
    axum::extract::ConnectInfo(_addr): axum::extract::ConnectInfo<std::net::SocketAddr>,
    state: axum::extract::State<AppState>,
    _req: axum::extract::Query<()>,
) -> impl IntoResponse {
    let modules: Vec<_> = state
        .modules
        .iter()
        .filter_map(|m| match m {
            Module::BatchedAsr { path, m } => {
                let config = m.config();
                let mut info = std::collections::HashMap::new();
                info.insert("type", "batched_asr".to_string());
                info.insert("path", path.to_string());
                info.insert("lm", config.lm_model_file.clone());
                info.insert("audio_tokenizer", config.audio_tokenizer_file.clone());
                info.insert("used_slots", m.used_slots().to_string());
                info.insert("total_slots", m.total_slots().to_string());
                Some(info)
            }
            _ => None,
        })
        .collect();
    utils::WrapJson(Ok(modules)).into_response()
}

#[derive(serde::Deserialize, serde::Serialize, Debug, Clone)]
struct AsrStreamingQuery {
    auth_id: Option<String>,
}

fn asr_router(s: Arc<asr::Asr>, path: &str, ss: &SharedState) -> axum::Router<()> {
    async fn asr_websocket(
        socket: axum::extract::ws::WebSocket,
        state: Arc<asr::Asr>,
        query: AsrStreamingQuery,
        _addr: Option<String>,
    ) {
        if let Err(err) = state.handle_socket(socket, query).await {
            tracing::error!(?err, "asr")
        }
    }

    async fn t(
        ws: axum::extract::ws::WebSocketUpgrade,
        headers: axum::http::HeaderMap,
        state: axum::extract::State<(Arc<asr::Asr>, SharedState)>,
        req: axum::extract::Query<AsrStreamingQuery>,
    ) -> utils::AxumResult<axum::response::Response> {
        let addr = headers.get("X-Real-IP").and_then(|v| v.to_str().ok().map(|v| v.to_string()));
        tracing::info!(addr, "handling asr-streaming query");
        // It's tricky to set the headers of a websocket in javascript so we pass the token via the
        // query too.
        let asr_query = req.0;
        let asr = state.0 .0.clone();
        let upg =
            ws.write_buffer_size(0).on_upgrade(move |v| asr_websocket(v, asr, asr_query, addr));
        Ok(upg)
    }
    axum::Router::new().route(path, axum::routing::get(t)).with_state((s, ss.clone()))
}

fn batched_asr_router(
    s: Arc<batched_asr::BatchedAsr>,
    path: &str,
    ss: &SharedState,
) -> axum::Router<()> {
    async fn asr_websocket(
        socket: axum::extract::ws::WebSocket,
        state: Arc<batched_asr::BatchedAsr>,
        query: AsrStreamingQuery,
        _addr: Option<String>,
    ) {
        if let Err(err) = state.handle_socket(socket, query).await {
            tracing::error!(?err, "asr")
        }
    }

    // TODO: add a batch mode.
    async fn t(
        state: axum::extract::State<(Arc<batched_asr::BatchedAsr>, SharedState)>,
        _headers: axum::http::HeaderMap,
        req: axum::body::Bytes,
    ) -> utils::AxumResult<Response> {
        tracing::info!(len = req.len(), "handling asr post query");
        let transcript = state.0 .0.handle_query(req).await?;
        Ok((
            StatusCode::OK,
            [(axum::http::header::CONTENT_TYPE, "application/json")],
            axum::Json(transcript),
        )
            .into_response())
    }

    async fn streaming_t(
        ws: axum::extract::ws::WebSocketUpgrade,
        headers: axum::http::HeaderMap,
        state: axum::extract::State<(Arc<batched_asr::BatchedAsr>, SharedState)>,
        req: axum::extract::Query<AsrStreamingQuery>,
    ) -> utils::AxumResult<axum::response::Response> {
        let addr = headers.get("X-Real-IP").and_then(|v| v.to_str().ok().map(|v| v.to_string()));
        tracing::info!(addr, "handling batched asr-streaming query");
        let asr_query = req.0;
        let asr = state.0 .0.clone();
        let upg =
            ws.write_buffer_size(0).on_upgrade(move |v| asr_websocket(v, asr, asr_query, addr));
        Ok(upg)
    }
    axum::Router::new()
        .route(path, axum::routing::post(t))
        .route(path, axum::routing::get(streaming_t))
        .with_state((s, ss.clone()))
}

// CR(laurent): tweak this comment.
// (carmen): Removed.
