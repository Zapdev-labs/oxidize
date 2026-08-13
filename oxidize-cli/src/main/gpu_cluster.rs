/// Handle the `oxidize gpu-cluster <subcommand>` family.
///
/// Subcommands:
///   generate [--family b200|h100|a100|rtx-pro-6000] [--nodes N] [--gpus-per-node N]
///            Emit the Kubernetes/Helm manifests from the GPU cluster spec.
///   detect   Probe the local node for NVIDIA GPUs via nvidia-smi.
///   profiles List the known GPU tier profiles.
pub(super) fn run_gpu_cluster(args: &[String]) -> i32 {
    use oxidize_core::gpu_cluster as gc;

    let sub = args.first().map(String::as_str).unwrap_or("help");
    match sub {
        "profiles" => {
            for p in gc::all_profiles() {
                println!(
                    "{:<14} product={:<26} arch={:<9} mem={}MiB tdp={}W nvlink={} mig={} timeslice={} net={}",
                    p.family.slug(),
                    p.product,
                    p.generation,
                    p.memory_mib,
                    p.tdp_watts,
                    p.nvlink,
                    p.mig_capable,
                    p.time_slice_replicas,
                    p.network_class,
                );
            }
            0
        }
        "detect" => {
            let gpus = gc::detect_gpus();
            if gpus.is_empty() {
                println!("no NVIDIA GPUs detected (nvidia-smi unavailable or no devices)");
                return 0;
            }
            for g in &gpus {
                let fam = g.family.map(|f| f.slug()).unwrap_or("unknown");
                println!(
                    "GPU {}: {} ({}MiB) family={} mig={}",
                    g.index, g.name, g.memory_total_mib, fam, g.mig_enabled
                );
            }
            println!("--- summary ---");
            for (fam, n) in gc::summarize(&gpus) {
                println!("{}: {}", fam.slug(), n);
            }
            0
        }
        "generate" => {
            let mut family: Option<gc::GpuFamily> = None;
            let mut nodes: u32 = 0;
            let mut gpus_per_node: u32 = 0;
            let mut i = 1;
            while i < args.len() {
                match args[i].as_str() {
                    "--family" => {
                        i += 1;
                        match args.get(i).and_then(|v| gc::GpuFamily::from_slug(v)) {
                            Some(f) => family = Some(f),
                            None => {
                                eprintln!("error: --family expects b200|h100|a100|rtx-pro-6000");
                                return 2;
                            }
                        }
                    }
                    "--nodes" => {
                        i += 1;
                        match args.get(i) {
                            Some(v) => match v.parse() {
                                Ok(n) => nodes = n,
                                Err(_) => {
                                    eprintln!(
                                        "error: --nodes expects a positive integer, got '{v}'"
                                    );
                                    return 2;
                                }
                            },
                            None => {
                                eprintln!("error: --nodes requires a value");
                                return 2;
                            }
                        }
                    }
                    "--gpus-per-node" => {
                        i += 1;
                        match args.get(i) {
                            Some(v) => match v.parse() {
                                Ok(n) => gpus_per_node = n,
                                Err(_) => {
                                    eprintln!(
                                        "error: --gpus-per-node expects a positive integer, got '{v}'"
                                    );
                                    return 2;
                                }
                            },
                            None => {
                                eprintln!("error: --gpus-per-node requires a value");
                                return 2;
                            }
                        }
                    }
                    other => {
                        eprintln!("error: unknown flag {other}");
                        return 2;
                    }
                }
                i += 1;
            }

            // Default to the full three-tier cluster from the spec when no
            // single family is selected.
            let specs = match family {
                Some(f) => {
                    let count = if nodes > 0 {
                        nodes
                    } else {
                        default_node_count(f)
                    };
                    let gpn = if gpus_per_node > 0 {
                        gpus_per_node
                    } else {
                        default_gpus_per_node(f)
                    };
                    vec![gc::NodePoolSpec::new(f, count, gpn)]
                }
                None => vec![
                    gc::NodePoolSpec::new(gc::GpuFamily::B200, 8, 8),
                    gc::NodePoolSpec::new(gc::GpuFamily::H100, 4, 8),
                    gc::NodePoolSpec::new(gc::GpuFamily::A100, 16, 8),
                    gc::NodePoolSpec::new(gc::GpuFamily::RtxPro6000, 4, 2),
                ],
            };
            let families: Vec<gc::GpuFamily> = specs.iter().map(|s| s.family).collect();

            print!("{}", gc::node_pools_yaml(&specs));
            println!("---");
            print!("{}", gc::device_plugin_config_yaml(&families));
            for f in &families {
                if let Some(mig) = gc::mig_config_yaml(*f) {
                    println!("---");
                    print!("{mig}");
                }
            }
            println!("---");
            print!("{}", gc::prometheus_rules_yaml());
            for f in &families {
                println!("---");
                print!("{}", gc::helm_values_yaml(*f));
            }
            0
        }
        _ => {
            eprintln!(
                "usage: oxidize gpu-cluster <generate|detect|profiles>\n\
                 \n\
                 generate [--family b200|h100|a100|rtx-pro-6000] [--nodes N] [--gpus-per-node N]\n\
                 detect   probe local NVIDIA GPUs via nvidia-smi\n\
                 profiles list known GPU tier profiles"
            );
            1
        }
    }
}

pub(super) fn default_node_count(f: oxidize_core::gpu_cluster::GpuFamily) -> u32 {
    use oxidize_core::gpu_cluster::GpuFamily::*;
    match f {
        B200 => 8,
        H100 => 4,
        A100 => 16,
        RtxPro6000 => 4,
    }
}

pub(super) fn default_gpus_per_node(f: oxidize_core::gpu_cluster::GpuFamily) -> u32 {
    use oxidize_core::gpu_cluster::GpuFamily::*;
    match f {
        B200 | H100 | A100 => 8,
        RtxPro6000 => 2,
    }
}
