use super::*;

pub(super) fn is_profiling_child() -> bool {
    std::env::var_os(PROFILE_CHILD_ENV).is_some()
}

pub(super) fn current_args_without_profile_flags() -> Vec<String> {
    filter_passthrough_args(std::env::args().skip(1))
}

pub(super) fn filter_passthrough_args<I>(input: I) -> Vec<String>
where
    I: IntoIterator<Item = String>,
{
    let mut filtered = Vec::new();
    let mut args = input.into_iter().peekable();
    while let Some(arg) = args.next() {
        let remove_next = arg == "--profile" || arg == "--profile-output";
        let remove_current =
            remove_next || arg.starts_with("--profile=") || arg.starts_with("--profile-output=");
        if !remove_current {
            filtered.push(arg);
        }
        if remove_next {
            let _ = args.next();
        }
    }
    filtered
}

pub(super) fn profiler_command(
    profiler: Profiler,
    output: Option<&PathBuf>,
    exe: &PathBuf,
    passthrough_args: &[String],
) -> Command {
    let mut command = match profiler {
        Profiler::Perf => {
            let mut cmd = Command::new("perf");
            cmd.arg("record").arg("--call-graph=dwarf");
            if let Some(path) = output {
                cmd.arg("-o").arg(path);
            }
            cmd
        }
        Profiler::Samply => {
            let mut cmd = Command::new("samply");
            cmd.arg("record");
            if let Some(path) = output {
                cmd.arg("-o").arg(path);
            }
            cmd
        }
    };
    command.env(PROFILE_CHILD_ENV, "1");
    command.arg(exe).args(passthrough_args);
    command
}

pub(super) fn run_profiled_inference(
    profiler: Profiler,
    output: Option<&PathBuf>,
) -> io::Result<ExitStatus> {
    let exe = std::env::current_exe()?;
    let passthrough_args = current_args_without_profile_flags();
    let mut command = profiler_command(profiler, output, &exe, &passthrough_args);
    command.status()
}

pub(super) fn run_api_server_blocking(server_args: oxidize_server::Args) -> io::Result<()> {
    let rt = tokio::runtime::Runtime::new()
        .map_err(|error| io::Error::other(format!("tokio runtime: {error}")))?;
    rt.block_on(async move {
        let (effective_backend, warning) = server_args.backend.to_core_backend().effective();
        if let Some(msg) = warning {
            eprintln!("warning: {msg}");
        }
        eprintln!(
            "server: loading model={} backend={} addr={}:{}",
            server_args
                .model
                .as_ref()
                .map(|path| path.display().to_string())
                .unwrap_or_else(|| "<none>".to_string()),
            effective_backend.as_str(),
            server_args.host,
            server_args.port
        );
        let loaded_model =
            oxidize_server::load_model_runtime_with_plan(&server_args).map_err(|error| {
                io::Error::other(format!("failed to initialize server model: {error}"))
            })?;
        let batch_mode =
            oxidize_server::effective_batch_mode(&server_args, loaded_model.autotune_plan.as_ref());
        let (model, paged) = if batch_mode == oxidize_server::BatchMode::Paged {
            if let Some(runtime) = loaded_model.runtime {
                (
                    None,
                    Some(oxidize_server::build_paged_runtime(
                        &server_args,
                        runtime,
                        loaded_model.autotune_plan.as_ref(),
                    )),
                )
            } else {
                (None, None)
            }
        } else {
            (loaded_model.runtime, None)
        };
        let api_key = std::env::var("OXIDIZE_API_KEY")
            .ok()
            .filter(|value| !value.is_empty());
        let state = oxidize_server::AppState {
            limiter: Arc::new(oxidize_server::RequestLimiter::new(
                oxidize_server::RequestLimitConfig::default(),
            )),
            batcher: Arc::new(oxidize_server::ContinuousBatcher::default()),
            auth: api_key
                .map(|key| oxidize_server::AuthConfig::from_keys([key]))
                .unwrap_or_else(oxidize_server::AuthConfig::disabled),
            model,
            paged,
            mesh: None,
            audit: Arc::new(oxidize_server::audit::AuditLogger::new()),
            metrics: Arc::new(
                oxidize_server::metrics::MetricsRegistry::new()
                    .map_err(|error| io::Error::other(format!("metrics registry: {error}")))?,
            ),
        };
        let app = oxidize_server::build_app_with_state(state);
        let listener =
            tokio::net::TcpListener::bind(SocketAddr::new(server_args.host, server_args.port))
                .await
                .map_err(|error| io::Error::other(format!("failed to bind server: {error}")))?;
        eprintln!(
            "server: listening on http://{}:{} (REST /v1/*, WebSocket ws://{}:{}/v1/realtime)",
            server_args.host, server_args.port, server_args.host, server_args.port
        );
        let shutdown_signal = oxidize_server::shutdown::ShutdownSignal::new();
        oxidize_server::shutdown::serve_with_graceful_shutdown(listener, app, shutdown_signal)
            .await;
        Ok(())
    })
}

pub(super) fn spawn_api_server_background(args: &Args) -> io::Result<()> {
    if args.model.is_none() {
        return Ok(());
    }
    let server_args = server_args_from_cli(args)?;
    let host = server_args.host;
    let port = server_args.port;
    std::thread::Builder::new()
        .name("oxidize-api".into())
        .spawn(move || {
            if let Err(error) = run_api_server_blocking(server_args) {
                eprintln!("api server failed: {error}");
            }
        })?;
    eprintln!(
        "api server: starting in background at http://{}:{} (REST /v1/*, WebSocket /v1/realtime)",
        host, port
    );
    Ok(())
}

pub(super) fn server_backend_from_cli(backend: Backend) -> oxidize_server::Backend {
    match backend {
        Backend::Cpu => oxidize_server::Backend::Cpu,
        Backend::Metal => oxidize_server::Backend::Metal,
        Backend::Mlx => oxidize_server::Backend::Mlx,
        Backend::Cuda => oxidize_server::Backend::Cuda,
        Backend::Rocm => oxidize_server::Backend::Rocm,
        Backend::Vulkan => oxidize_server::Backend::Vulkan,
        Backend::IntelArc => oxidize_server::Backend::IntelArc,
    }
}

pub(super) fn server_args_from_cli(args: &Args) -> io::Result<oxidize_server::Args> {
    let host = args
        .api_host
        .parse::<IpAddr>()
        .map_err(|error| io::Error::other(format!("invalid --host/--api-host: {error}")))?;
    let model_id = args
        .model
        .as_ref()
        .and_then(|path| path.file_stem())
        .and_then(|stem| stem.to_str())
        .unwrap_or("oxidize-default")
        .to_string();
    Ok(oxidize_server::Args {
        host,
        port: args.api_port,
        model: args.model.clone(),
        backend: server_backend_from_cli(args.backend),
        batch_mode: oxidize_server::BatchMode::Sequential,
        model_id,
        max_tokens: args.max_tokens,
        temperature: args.temperature,
        top_p: args.top_p,
        top_k: args.top_k,
        ctx_size: args.ctx_size,
        prefill_batch_size: 512,
        prefill_chunk_size: 16,
        cpu_optimized: args.cpu_optimized,
        ram_offload: args.ram_offload,
        mmap_prefetch: args.mmap_prefetch,
        mmap_hugepages: args.mmap_hugepages,
        layer_wise: args.layer_wise,
        layer_cache: args.layer_cache,
        turboquant_kv: args.turboquant,
        no_turboquant_kv: args.no_turboquant,
        mesh: args.mesh,
        mesh_port: args.mesh_port,
        tokenizer_model: args.tokenizer_model.clone(),
        draft_model: args.draft_model.clone(),
        draft_tokens: args.draft_tokens,
        kv_cache_dtype: match args.kv_cache_dtype {
            KvCacheDType::F32 => oxidize_server::KvCacheDType::F32,
            KvCacheDType::F16 => oxidize_server::KvCacheDType::F16,
            KvCacheDType::Q8 => oxidize_server::KvCacheDType::Q8,
            KvCacheDType::Q4 => oxidize_server::KvCacheDType::Q4,
        },
        threads: args.threads.filter(|threads| *threads > 0).unwrap_or(0),
        ram_offload_threads: args.ram_offload_threads,
        auto: args.auto,
        no_auto: args.no_auto,
        print_plan: args.print_plan.clone(),
    })
}

pub(super) fn run_api_server_in_process(args: &Args) -> io::Result<()> {
    run_api_server_blocking(server_args_from_cli(args)?)
}

/// Run the CLI in distributed mesh node mode.
/// Delegates to `oxidize_core::mesh::run_mesh_node` which builds the
/// libp2p swarm, starts mDNS, subscribes to all 6 GossipSub topics, and
/// drives the event loop.
pub(super) fn run_mesh_mode(mesh_port: u16) -> io::Result<()> {
    let rt = tokio::runtime::Runtime::new()
        .map_err(|e| io::Error::other(format!("tokio runtime: {e}")))?;
    rt.block_on(async {
        oxidize_core::mesh::run_mesh_node(mesh_port, None, None, None)
            .await
            .map_err(|e| io::Error::other(format!("mesh node error: {e}")))
    })
}
