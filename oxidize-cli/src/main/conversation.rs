use super::*;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct ConversationTurn {
    user: String,
    assistant: String,
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub(super) struct ConversationHistory {
    turns: Vec<ConversationTurn>,
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub(super) struct PromptCache {
    responses: HashMap<String, String>,
}

impl PromptCache {
    pub(super) fn get(&self, prompt: &str) -> Option<&str> {
        self.responses.get(prompt).map(String::as_str)
    }

    pub(super) fn insert(&mut self, prompt: &str, response: &str) {
        self.responses
            .insert(prompt.to_owned(), response.to_owned());
    }
}

impl ConversationHistory {
    pub(super) fn add_turn(&mut self, user: &str, assistant: &str) {
        self.turns.push(ConversationTurn {
            user: user.to_owned(),
            assistant: assistant.to_owned(),
        });
    }

    pub(super) fn clear(&mut self) {
        self.turns.clear();
    }

    pub(super) fn render(&self) -> String {
        if self.turns.is_empty() {
            return "no conversation history".to_owned();
        }

        self.turns
            .iter()
            .enumerate()
            .map(|(index, turn)| {
                format!(
                    "{}. user: {}\n   assistant: {}",
                    index + 1,
                    turn.user,
                    turn.assistant
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    }
}

pub(super) fn read_chat_prompt<R: BufRead, W: Write>(
    reader: &mut R,
    writer: &mut W,
) -> io::Result<Option<String>> {
    let mut lines = Vec::new();
    loop {
        let mut input = String::new();
        if reader.read_line(&mut input)? == 0 {
            if lines.is_empty() {
                return Ok(None);
            }
            break;
        }

        let trimmed = input.trim_end_matches(['\r', '\n']);
        let continues = trimmed.ends_with('\\');
        let line = if continues {
            trimmed[..trimmed.len() - 1].to_owned()
        } else {
            trimmed.to_owned()
        };
        lines.push(line);

        if continues {
            write!(writer, "| ")?;
            writer.flush()?;
            continue;
        }
        break;
    }
    Ok(Some(lines.join("\n")))
}

pub(super) fn run_chat_mode<R: BufRead, W: Write>(
    reader: &mut R,
    writer: &mut W,
) -> io::Result<()> {
    writeln!(writer, "oxidize-cli chat mode. type 'exit' to quit.")?;
    let mut history = ConversationHistory::default();
    let mut prompt_cache = PromptCache::default();
    loop {
        write!(writer, "> ")?;
        writer.flush()?;

        let Some(input) = read_chat_prompt(reader, writer)? else {
            break;
        };

        let prompt = input.trim();
        if prompt.eq_ignore_ascii_case("exit") || prompt.eq_ignore_ascii_case("quit") {
            writeln!(writer, "bye")?;
            break;
        }
        if prompt.is_empty() {
            continue;
        }

        if prompt.eq_ignore_ascii_case("/history") {
            writeln!(writer, "{}", history.render())?;
            continue;
        }
        if prompt.eq_ignore_ascii_case("/clear") {
            history.clear();
            writeln!(writer, "conversation history cleared")?;
            continue;
        }

        let response = write_generated_response_cached(prompt, &mut prompt_cache, writer)?;
        history.add_turn(prompt, &response);
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn run_model_chat_mode<R: BufRead, W: Write, M: Model>(
    reader: &mut R,
    writer: &mut W,
    model: &mut M,
    tokenizer: &LoadedTokenizer,
    max_tokens: usize,
    temperature: f32,
    top_p: Option<f32>,
    top_k: Option<usize>,
) -> io::Result<()> {
    writeln!(
        writer,
        "╭─ oxidize chat ─────────────────────────────────────╮"
    )?;
    writeln!(
        writer,
        "│ API server is running if enabled; type /exit to quit │"
    )?;
    writeln!(
        writer,
        "│ multiline: end a line with \\                         │"
    )?;
    writeln!(
        writer,
        "╰──────────────────────────────────────────────────────╯"
    )?;
    let mut history = ConversationHistory::default();
    loop {
        write!(writer, "\nYou › ")?;
        writer.flush()?;
        let Some(input) = read_chat_prompt(reader, writer)? else {
            break;
        };
        let prompt = input.trim();
        if prompt.eq_ignore_ascii_case("/exit")
            || prompt.eq_ignore_ascii_case("exit")
            || prompt.eq_ignore_ascii_case("quit")
        {
            writeln!(writer, "bye")?;
            break;
        }
        if prompt.eq_ignore_ascii_case("/history") {
            writeln!(writer, "{}", history.render())?;
            continue;
        }
        if prompt.eq_ignore_ascii_case("/clear") {
            history.clear();
            writeln!(writer, "conversation history cleared")?;
            continue;
        }
        if prompt.is_empty() {
            continue;
        }
        writeln!(writer, "\nAssistant ›")?;
        let response = generate_with_model(
            prompt,
            model,
            tokenizer,
            max_tokens,
            temperature,
            top_p,
            top_k,
            writer,
        )?;
        history.add_turn(prompt, &response);
    }
    Ok(())
}

/// Run the CLI in mesh + chat combined mode.
///
/// Starts a background mesh node, waits for leader election, then opens an
/// interactive REPL.  Each user prompt is broadcast to the mesh master via
/// GossipSub `COMMANDS` and response tokens stream back through the mesh data
/// plane (real or local fallback) and are printed token-by-token.
pub(super) fn run_mesh_chat_mode(mesh_port: u16) -> io::Result<()> {
    use oxidize_core::mesh::{MeshChatPrompt, MeshChatToken};

    let rt = tokio::runtime::Runtime::new()
        .map_err(|e| io::Error::other(format!("tokio runtime: {e}")))?;

    let (prompt_tx, prompt_rx) = tokio::sync::mpsc::unbounded_channel::<MeshChatPrompt>();
    let (token_tx, mut token_rx) = tokio::sync::mpsc::unbounded_channel::<MeshChatToken>();

    // Spawn the mesh node in a background task within the same runtime.
    let mesh_handle = rt.spawn(async move {
        let result =
            oxidize_core::mesh::run_mesh_node(mesh_port, None, Some(prompt_rx), Some(token_tx))
                .await;
        if let Err(ref e) = result {
            eprintln!("mesh node error: {e}");
        }
        result
    });

    // Give the mesh node a moment to start up and discover peers.
    rt.block_on(async {
        tokio::time::sleep(Duration::from_secs(2)).await;
    });

    let stdin = io::stdin();
    let mut reader = stdin.lock();
    let stdout = io::stdout();
    let mut writer = stdout.lock();

    writeln!(writer, "oxidize-cli mesh chat mode. type 'exit' to quit.")?;
    let mut history = ConversationHistory::default();
    let mut prompt_cache = PromptCache::default();
    let mut request_counter: usize = 0;

    loop {
        write!(writer, "> ")?;
        writer.flush()?;

        let Some(input) = read_chat_prompt(&mut reader, &mut writer)? else {
            break;
        };

        let prompt = input.trim();
        if prompt.eq_ignore_ascii_case("exit") || prompt.eq_ignore_ascii_case("quit") {
            writeln!(writer, "bye")?;
            break;
        }
        if prompt.is_empty() {
            continue;
        }
        if prompt.eq_ignore_ascii_case("/history") {
            writeln!(writer, "{}", history.render())?;
            continue;
        }
        if prompt.eq_ignore_ascii_case("/clear") {
            history.clear();
            writeln!(writer, "conversation history cleared")?;
            continue;
        }

        let cached = prompt_cache.get(prompt);
        let response = if let Some(cached) = cached {
            writeln!(writer, "{cached}")?;
            writeln!(
                writer,
                "generation stats: tokens=0 speed=0.00 tok/s (cache hit)"
            )?;
            cached.to_owned()
        } else {
            request_counter += 1;
            let request_id = format!("cli-{}", request_counter);

            // Broadcast the prompt to the mesh via the chat engine.
            let mesh_prompt = MeshChatPrompt {
                request_id: request_id.clone(),
                prompt: prompt.to_string(),
                max_tokens: 8, // short deterministic tokens for the demo
                temperature: 0.0,
                top_p: 0.0,
            };
            if prompt_tx.send(mesh_prompt).is_err() {
                writeln!(writer, "mesh prompt channel closed")?;
                break;
            }

            // Drain streaming tokens from the mesh data plane.
            let mut token_count = 0usize;
            let mut response_parts = Vec::new();
            let start = Instant::now();

            // Give the master a moment to receive the prompt and start generating.
            std::thread::sleep(Duration::from_millis(100));

            loop {
                match token_rx.try_recv() {
                    Ok(token) => {
                        if token.request_id == request_id {
                            write!(writer, "{}", token.token)?;
                            writer.flush()?;
                            token_count += 1;
                            response_parts.push(token.token);
                            if token.is_final {
                                writeln!(writer)?;
                                break;
                            } else {
                                write!(writer, " ")?;
                            }
                            // Tiny artificial pacing so the TUI shows progress.
                            std::thread::sleep(Duration::from_millis(20));
                        }
                    }
                    Err(tokio::sync::mpsc::error::TryRecvError::Empty) => {
                        // Poll briefly then give up if nothing arrives.
                        std::thread::sleep(Duration::from_millis(50));
                        if start.elapsed() > Duration::from_secs(5) {
                            // Timeout waiting for tokens.
                            break;
                        }
                    }
                    Err(tokio::sync::mpsc::error::TryRecvError::Disconnected) => {
                        writeln!(writer, "mesh token channel disconnected")?;
                        break;
                    }
                }
            }

            let elapsed = start.elapsed().as_secs_f64();
            let speed = if elapsed > 0.0 {
                token_count as f64 / elapsed
            } else {
                0.0
            };
            writeln!(
                writer,
                "generation stats: tokens={token_count} speed={speed:.2} tok/s (mesh)"
            )?;
            let response = response_parts.join(" ");
            prompt_cache.insert(prompt, &response);
            response
        };

        history.add_turn(prompt, &response);
    }

    // Abort the mesh node when chat exits.
    mesh_handle.abort();
    let _ = rt.block_on(mesh_handle);
    Ok(())
}
