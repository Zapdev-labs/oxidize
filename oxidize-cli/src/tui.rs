//! Full-screen console ported from the former TypeScript `oxidize-tui`.
//! Drives `oxidize serve` over HTTP (or `--api`) instead of linking weights.

use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use crossterm::terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen};
use crossterm::ExecutableCommand;
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, Gauge, List, ListItem, Paragraph, Tabs, Wrap};
use ratatui::{Frame, Terminal};
use serde_json::Value;
use std::collections::HashMap;
use std::fs;
use std::io::{self, BufRead, BufReader, Read};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread;
use std::time::{Duration, Instant};

#[derive(Clone)]
pub struct TuiOpts {
    pub model: Option<PathBuf>,
    pub api: Option<String>,
    pub backend: String,
    pub threads: usize,
    pub ctx_size: usize,
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum View {
    Chat,
    Models,
    Monitor,
    Logs,
}

impl View {
    fn all() -> [View; 4] {
        [View::Chat, View::Models, View::Monitor, View::Logs]
    }
    fn title(self) -> &'static str {
        match self {
            View::Chat => "chat",
            View::Models => "models",
            View::Monitor => "monitor",
            View::Logs => "logs",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ServerStatus {
    Idle,
    Starting,
    Ready,
    Error,
    Stopped,
}

#[derive(Clone)]
struct ChatMsg {
    role: &'static str,
    content: String,
    error: bool,
}

#[derive(Clone)]
struct ModelRow {
    path: PathBuf,
    name: String,
    size: u64,
    source: String,
    facts: String,
}

#[derive(Clone, Default)]
struct Metrics {
    tokens_per_second: f64,
    tokens_generated: f64,
    requests_in_flight: f64,
    requests_total: f64,
    queue_depth: f64,
    kv_cache_bytes: f64,
    errors_total: f64,
    error: Option<String>,
}

struct Overlay {
    kind: OverlayKind,
    query: String,
    cursor: usize,
    label: String,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum OverlayKind {
    Palette,
    Help,
    Prompt,
}

struct App {
    view: View,
    status: ServerStatus,
    url: Option<String>,
    model_id: String,
    model_path: Option<PathBuf>,
    detail: String,
    external: bool,
    backend: String,
    threads: usize,
    ctx_size: usize,
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
    top_k: usize,
    draft: String,
    messages: Vec<ChatMsg>,
    generating: bool,
    tps: f64,
    models: Vec<ModelRow>,
    model_cursor: usize,
    model_filter: String,
    filtering: bool,
    logs: Vec<String>,
    log_follow: bool,
    metrics: Metrics,
    overlay: Option<Overlay>,
    toast: Option<(String, Instant)>,
    child: Option<Child>,
    events: Receiver<UiEvent>,
    tx: Sender<UiEvent>,
}

enum UiEvent {
    Log(String),
    Ready { url: String, model_id: String },
    Failed(String),
    Delta(String),
    ChatDone,
    ChatErr(String),
    Metrics(Metrics),
    Models(Vec<ModelRow>),
}

pub fn run(opts: TuiOpts) -> io::Result<()> {
    let (tx, rx) = mpsc::channel();
    let mut app = App {
        view: if opts.model.is_some() || opts.api.is_some() {
            View::Chat
        } else {
            View::Models
        },
        status: ServerStatus::Idle,
        url: None,
        model_id: "oxidize-default".into(),
        model_path: opts.model.clone(),
        detail: String::new(),
        external: opts.api.is_some(),
        backend: opts.backend,
        threads: opts.threads,
        ctx_size: opts.ctx_size,
        max_tokens: opts.max_tokens,
        temperature: opts.temperature,
        top_p: opts.top_p,
        top_k: opts.top_k,
        draft: String::new(),
        messages: Vec::new(),
        generating: false,
        tps: 0.0,
        models: Vec::new(),
        model_cursor: 0,
        model_filter: String::new(),
        filtering: false,
        logs: Vec::new(),
        log_follow: true,
        metrics: Metrics::default(),
        overlay: None,
        toast: None,
        child: None,
        events: rx,
        tx: tx.clone(),
    };

    scan_models_async(tx.clone());
    if let Some(url) = opts.api {
        app.attach(&url);
    } else if let Some(path) = opts.model {
        app.load_model(&path);
    }

    enable_raw_mode()?;
    let mut stdout = io::stdout();
    stdout.execute(EnterAlternateScreen)?;
    let mut terminal = Terminal::new(CrosstermBackend::new(stdout))?;
    let result = app_loop(&mut terminal, &mut app);
    disable_raw_mode()?;
    io::stdout().execute(LeaveAlternateScreen)?;
    app.shutdown();
    result
}

fn app_loop(terminal: &mut Terminal<CrosstermBackend<io::Stdout>>, app: &mut App) -> io::Result<()> {
    loop {
        while let Ok(ev) = app.events.try_recv() {
            app.handle_event(ev);
        }
        terminal.draw(|f| draw(f, app))?;
        if !event::poll(Duration::from_millis(40))? {
            continue;
        }
        let Event::Key(key) = event::read()? else {
            continue;
        };
        if key.kind != KeyEventKind::Press {
            continue;
        }
        if app.handle_key(key) {
            break;
        }
    }
    Ok(())
}

impl App {
    fn log(&mut self, line: impl Into<String>) {
        self.logs.push(line.into());
        if self.logs.len() > 2000 {
            self.logs.drain(0..self.logs.len() - 2000);
        }
    }

    fn toast(&mut self, text: impl Into<String>) {
        self.toast = Some((text.into(), Instant::now()));
    }

    fn filtered_models(&self) -> Vec<&ModelRow> {
        self.models
            .iter()
            .filter(|m| fuzzy(&self.model_filter, &m.name) || fuzzy(&self.model_filter, &m.path.to_string_lossy()))
            .collect()
    }

    fn attach(&mut self, url: &str) {
        let clean = url.trim_end_matches('/').to_string();
        self.status = ServerStatus::Starting;
        self.external = true;
        self.url = Some(clean.clone());
        self.detail = format!("attaching to {clean}");
        let tx = self.tx.clone();
        thread::spawn(move || attach_worker(clean, tx));
    }

    fn load_model(&mut self, path: &Path) {
        if self.external {
            self.toast("attached to an external server; detach first");
            return;
        }
        self.shutdown_child();
        self.status = ServerStatus::Starting;
        self.model_path = Some(path.to_path_buf());
        self.model_id = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("model")
            .to_string();
        self.view = View::Chat;
        self.detail = "spawning".into();
        self.log(format!("loading {}", path.display()));
        let tx = self.tx.clone();
        let exe = std::env::current_exe().ok();
        let path = path.to_path_buf();
        let backend = self.backend.clone();
        let threads = self.threads;
        let ctx = self.ctx_size;
        let max_tokens = self.max_tokens;
        match spawn_server(&path, exe.as_deref(), &backend, threads, ctx, max_tokens, tx.clone()) {
            Ok(child) => self.child = Some(child),
            Err(err) => {
                self.status = ServerStatus::Error;
                self.detail = err.to_string();
                self.log(err.to_string());
            }
        }
    }

    fn shutdown_child(&mut self) {
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
        self.status = ServerStatus::Stopped;
        self.url = None;
    }

    fn shutdown(&mut self) {
        self.shutdown_child();
    }

    fn send_chat(&mut self) {
        let text = self.draft.trim().to_string();
        if text.is_empty() || self.generating {
            return;
        }
        let Some(url) = self.url.clone() else {
            self.toast("no model loaded — press ctrl+t for the model browser");
            return;
        };
        if self.status != ServerStatus::Ready {
            self.toast("server not ready");
            return;
        }
        self.draft.clear();
        self.messages.push(ChatMsg {
            role: "user",
            content: text.clone(),
            error: false,
        });
        self.messages.push(ChatMsg {
            role: "assistant",
            content: String::new(),
            error: false,
        });
        self.generating = true;
        let mut history = Vec::new();
        for m in &self.messages {
            if m.role == "assistant" && m.content.is_empty() {
                continue;
            }
            history.push((m.role.to_string(), m.content.clone()));
        }
        let model = self.model_id.clone();
        let params = (self.temperature, self.top_p, self.top_k, self.max_tokens);
        let tx = self.tx.clone();
        thread::spawn(move || stream_chat(url, model, history, params, tx));
    }

    fn handle_event(&mut self, ev: UiEvent) {
        match ev {
            UiEvent::Log(line) => self.log(line),
            UiEvent::Ready { url, model_id } => {
                self.status = ServerStatus::Ready;
                self.url = Some(url.clone());
                self.model_id = model_id;
                self.detail.clear();
                self.toast("ready");
                self.log(format!("ready on {url}"));
                start_metrics(url, self.tx.clone());
            }
            UiEvent::Failed(msg) => {
                self.status = ServerStatus::Error;
                self.detail = msg.clone();
                self.toast(msg);
            }
            UiEvent::Delta(delta) => {
                if let Some(last) = self.messages.last_mut() {
                    if last.role == "assistant" {
                        last.content.push_str(&delta);
                    }
                }
            }
            UiEvent::ChatDone => self.generating = false,
            UiEvent::ChatErr(msg) => {
                self.generating = false;
                if let Some(last) = self.messages.last_mut() {
                    last.error = true;
                    if last.content.is_empty() {
                        last.content = msg;
                    } else {
                        last.content.push_str("\n\n[error] ");
                        last.content.push_str(&msg);
                    }
                }
            }
            UiEvent::Metrics(m) => {
                self.tps = m.tokens_per_second;
                self.metrics = m;
            }
            UiEvent::Models(rows) => {
                self.models = rows;
                if self.model_cursor >= self.models.len() {
                    self.model_cursor = self.models.len().saturating_sub(1);
                }
            }
        }
    }

    fn handle_key(&mut self, key: KeyEvent) -> bool {
        if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c') {
            return true;
        }
        if let Some(overlay) = self.overlay.as_mut() {
            match overlay.kind {
                OverlayKind::Help => {
                    if matches!(key.code, KeyCode::Esc | KeyCode::Enter | KeyCode::Char('q')) {
                        self.overlay = None;
                    }
                    return false;
                }
                OverlayKind::Palette => {
                    match key.code {
                        KeyCode::Esc => self.overlay = None,
                        KeyCode::Up => overlay.cursor = overlay.cursor.saturating_sub(1),
                        KeyCode::Down => overlay.cursor += 1,
                        KeyCode::Backspace => {
                            overlay.query.pop();
                            overlay.cursor = 0;
                        }
                        KeyCode::Char(c) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
                            overlay.query.push(c);
                            overlay.cursor = 0;
                        }
                        KeyCode::Enter => {
                            let query = overlay.query.clone();
                            let cursor = overlay.cursor;
                            self.overlay = None;
                            self.run_palette(&query, cursor);
                        }
                        _ => {}
                    }
                    return false;
                }
                OverlayKind::Prompt => {
                    match key.code {
                        KeyCode::Esc => self.overlay = None,
                        KeyCode::Backspace => {
                            overlay.query.pop();
                        }
                        KeyCode::Char(c) => overlay.query.push(c),
                        KeyCode::Enter => {
                            let repo = overlay.query.clone();
                            self.overlay = None;
                            self.view = View::Logs;
                            self.toast(format!("pull {repo} (use oxidize pull)"));
                            self.log(format!("palette pull requested: {repo}"));
                        }
                        _ => {}
                    }
                    return false;
                }
            }
        }

        if key.modifiers.contains(KeyModifiers::CONTROL) {
            match key.code {
                KeyCode::Char('k') => {
                    self.overlay = Some(Overlay {
                        kind: OverlayKind::Palette,
                        query: String::new(),
                        cursor: 0,
                        label: String::new(),
                    });
                    return false;
                }
                KeyCode::Char('t') => {
                    let all = View::all();
                    let i = all.iter().position(|v| *v == self.view).unwrap_or(0);
                    self.view = all[(i + 1) % all.len()];
                    return false;
                }
                KeyCode::Char('l') if self.view == View::Chat => {
                    self.messages.clear();
                    return false;
                }
                _ => {}
            }
        }

        if key.code == KeyCode::Esc {
            if self.generating {
                self.generating = false;
                self.toast("cancelled (in-flight stream may finish)");
            }
            self.filtering = false;
            return false;
        }

        if self.view == View::Chat {
            match key.code {
                KeyCode::Enter => self.send_chat(),
                KeyCode::Backspace => {
                    self.draft.pop();
                }
                KeyCode::Char(c) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
                    self.draft.push(c);
                }
                _ => {}
            }
            return false;
        }

        if self.view == View::Models && self.filtering {
            match key.code {
                KeyCode::Enter => self.filtering = false,
                KeyCode::Backspace => {
                    self.model_filter.pop();
                }
                KeyCode::Char(c) => self.model_filter.push(c),
                _ => {}
            }
            return false;
        }

        match key.code {
            KeyCode::Char('q') => return true,
            KeyCode::Char('?') => {
                self.overlay = Some(Overlay {
                    kind: OverlayKind::Help,
                    query: String::new(),
                    cursor: 0,
                    label: String::new(),
                });
            }
            KeyCode::Char('1') => self.view = View::Chat,
            KeyCode::Char('2') => self.view = View::Models,
            KeyCode::Char('3') => self.view = View::Monitor,
            KeyCode::Char('4') => self.view = View::Logs,
            _ => {}
        }

        if self.view == View::Models {
            let n = self.filtered_models().len();
            match key.code {
                KeyCode::Down | KeyCode::Char('j') => {
                    self.model_cursor = (self.model_cursor + 1).min(n.saturating_sub(1));
                }
                KeyCode::Up | KeyCode::Char('k') => {
                    self.model_cursor = self.model_cursor.saturating_sub(1);
                }
                KeyCode::Char('/') => {
                    self.filtering = true;
                    self.model_filter.clear();
                }
                KeyCode::Char('r') => scan_models_async(self.tx.clone()),
                KeyCode::Char('p') => {
                    self.overlay = Some(Overlay {
                        kind: OverlayKind::Prompt,
                        query: String::new(),
                        cursor: 0,
                        label: "pull from Hugging Face".into(),
                    });
                }
                KeyCode::Enter => {
                    let path = self
                        .filtered_models()
                        .get(self.model_cursor)
                        .map(|m| m.path.clone());
                    if let Some(path) = path {
                        self.load_model(&path);
                    }
                }
                _ => {}
            }
        }
        if self.view == View::Logs {
            if key.code == KeyCode::Char('f') {
                self.log_follow = !self.log_follow;
            }
            if key.code == KeyCode::Char('c') {
                self.logs.clear();
            }
        }
        false
    }

    fn run_palette(&mut self, query: &str, cursor: usize) {
        let cmds = palette_commands();
        let matches: Vec<_> = cmds.into_iter().filter(|c| fuzzy(query, c)).collect();
        let Some(cmd) = matches.get(cursor.min(matches.len().saturating_sub(1))) else {
            return;
        };
        match *cmd {
            "Go to chat" => self.view = View::Chat,
            "Go to models" => self.view = View::Models,
            "Go to monitor" => self.view = View::Monitor,
            "Go to logs" => self.view = View::Logs,
            "Clear conversation" => self.messages.clear(),
            "Rescan model directories" => scan_models_async(self.tx.clone()),
            "Stop server" => {
                self.shutdown_child();
                self.toast("server stopped");
            }
            "Restart server" => {
                if let Some(path) = self.model_path.clone() {
                    self.load_model(&path);
                }
            }
            "Temperature +0.1" => {
                self.temperature = (self.temperature + 0.1).min(2.0);
                self.toast(format!("temperature = {:.1}", self.temperature));
            }
            "Temperature -0.1" => {
                self.temperature = (self.temperature - 0.1).max(0.0);
                self.toast(format!("temperature = {:.1}", self.temperature));
            }
            _ => {}
        }
    }
}

fn palette_commands() -> Vec<&'static str> {
    vec![
        "Go to chat",
        "Go to models",
        "Go to monitor",
        "Go to logs",
        "Clear conversation",
        "Rescan model directories",
        "Restart server",
        "Stop server",
        "Temperature +0.1",
        "Temperature -0.1",
    ]
}

fn fuzzy(query: &str, hay: &str) -> bool {
    if query.is_empty() {
        return true;
    }
    hay.to_ascii_lowercase()
        .contains(&query.to_ascii_lowercase())
}

fn draw(f: &mut Frame, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Min(3),
            Constraint::Length(1),
            Constraint::Length(1),
        ])
        .split(f.area());

    let status = match app.status {
        ServerStatus::Idle => "idle",
        ServerStatus::Starting => "starting",
        ServerStatus::Ready => "ready",
        ServerStatus::Error => "error",
        ServerStatus::Stopped => "stopped",
    };
    let header = format!(
        " oxidize  {status}  {}  tps={:.1}  {}",
        app.model_id,
        app.tps,
        app.url.as_deref().unwrap_or("-")
    );
    f.render_widget(
        Paragraph::new(header).style(Style::default().fg(Color::Cyan)),
        chunks[0],
    );

    let titles: Vec<Line> = View::all().map(|v| Line::from(v.title())).to_vec();
    let selected = View::all()
        .iter()
        .position(|v| *v == app.view)
        .unwrap_or(0);
    f.render_widget(
        Tabs::new(titles)
            .select(selected)
            .highlight_style(Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
        chunks[1],
    );

    match app.view {
        View::Chat => draw_chat(f, chunks[2], app),
        View::Models => draw_models(f, chunks[2], app),
        View::Monitor => draw_monitor(f, chunks[2], app),
        View::Logs => draw_logs(f, chunks[2], app),
    }

    let hints = match app.view {
        View::Chat => "enter send  esc cancel  ctrl+t views  ctrl+k palette",
        View::Models => "enter load  / filter  r rescan  p pull  j/k move",
        View::Monitor => "ctrl+t views",
        View::Logs => "f follow  c clear",
    };
    f.render_widget(Paragraph::new(hints).style(Style::default().fg(Color::DarkGray)), chunks[3]);

    let footer = if let Some((text, at)) = &app.toast {
        if at.elapsed() < Duration::from_secs(3) {
            text.as_str()
        } else {
            app.detail.as_str()
        }
    } else {
        app.detail.as_str()
    };
    f.render_widget(Paragraph::new(footer), chunks[4]);

    if let Some(overlay) = &app.overlay {
        draw_overlay(f, overlay);
    }
}

fn draw_chat(f: &mut Frame, area: Rect, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(1), Constraint::Length(3)])
        .split(area);
    let lines: Vec<Line> = app
        .messages
        .iter()
        .flat_map(|m| {
            let color = if m.error {
                Color::Red
            } else if m.role == "user" {
                Color::Green
            } else {
                Color::White
            };
            vec![
                Line::from(Span::styled(m.role, Style::default().fg(color).add_modifier(Modifier::BOLD))),
                Line::from(m.content.clone()),
                Line::from(""),
            ]
        })
        .collect();
    f.render_widget(
        Paragraph::new(lines)
            .wrap(Wrap { trim: false })
            .block(Block::default().borders(Borders::ALL).title("transcript")),
        chunks[0],
    );
    f.render_widget(
        Paragraph::new(app.draft.as_str())
            .block(Block::default().borders(Borders::ALL).title("prompt")),
        chunks[1],
    );
}

fn draw_models(f: &mut Frame, area: Rect, app: &App) {
    let rows = app.filtered_models();
    let items: Vec<ListItem> = rows
        .iter()
        .enumerate()
        .map(|(i, m)| {
            let mark = if i == app.model_cursor { ">" } else { " " };
            ListItem::new(format!(
                "{mark} {:8} {:10}  {}  {}",
                m.source,
                format_bytes(m.size),
                m.facts,
                m.name
            ))
        })
        .collect();
    let title = if app.filtering {
        format!("models /{}", app.model_filter)
    } else {
        format!("models ({})", rows.len())
    };
    f.render_widget(
        List::new(items).block(Block::default().borders(Borders::ALL).title(title)),
        area,
    );
}

fn draw_monitor(f: &mut Frame, area: Rect, app: &App) {
    let m = &app.metrics;
    let text = format!(
        "tokens/s        {:>10.2}\n\
         generated       {:>10.0}\n\
         in flight       {:>10.0}\n\
         requests        {:>10.0}\n\
         queue           {:>10.0}\n\
         kv cache        {:>10}\n\
         errors          {:>10.0}\n\
         {}",
        m.tokens_per_second,
        m.tokens_generated,
        m.requests_in_flight,
        m.requests_total,
        m.queue_depth,
        format_bytes(m.kv_cache_bytes as u64),
        m.errors_total,
        m.error.clone().unwrap_or_default()
    );
    let ratio = (m.tokens_per_second / 100.0).clamp(0.0, 1.0) as f64;
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(3), Constraint::Min(1)])
        .split(area);
    f.render_widget(
        Gauge::default()
            .block(Block::default().borders(Borders::ALL).title("throughput"))
            .ratio(ratio)
            .label(format!("{:.1} tok/s", m.tokens_per_second)),
        chunks[0],
    );
    f.render_widget(
        Paragraph::new(text).block(Block::default().borders(Borders::ALL).title("/metrics")),
        chunks[1],
    );
}

fn draw_logs(f: &mut Frame, area: Rect, app: &App) {
    let start = if app.log_follow {
        app.logs.len().saturating_sub(area.height.saturating_sub(2) as usize)
    } else {
        0
    };
    let text = app.logs.get(start..).unwrap_or(&[]).join("\n");
    f.render_widget(
        Paragraph::new(text).block(Block::default().borders(Borders::ALL).title("logs")),
        area,
    );
}

fn draw_overlay(f: &mut Frame, overlay: &Overlay) {
    let area = centered(f.area(), 70, 16);
    f.render_widget(Clear, area);
    match overlay.kind {
        OverlayKind::Help => {
            let help = "ctrl+k palette   ctrl+t next view   1-4 jump views\n\
esc cancel   q quit (not in chat)   enter send/load\n\
models: j/k move  / filter  r rescan  p pull";
            f.render_widget(
                Paragraph::new(help).block(Block::default().borders(Borders::ALL).title("help")),
                area,
            );
        }
        OverlayKind::Palette => {
            let cmds = palette_commands();
            let matches: Vec<_> = cmds.into_iter().filter(|c| fuzzy(&overlay.query, c)).collect();
            let cursor = overlay.cursor.min(matches.len().saturating_sub(1));
            let body: String = matches
                .iter()
                .enumerate()
                .map(|(i, c)| format!("{} {c}", if i == cursor { ">" } else { " " }))
                .collect::<Vec<_>>()
                .join("\n");
            f.render_widget(
                Paragraph::new(format!("> {}\n{body}", overlay.query))
                    .block(Block::default().borders(Borders::ALL).title("palette")),
                area,
            );
        }
        OverlayKind::Prompt => {
            f.render_widget(
                Paragraph::new(format!("{}: {}", overlay.label, overlay.query))
                    .block(Block::default().borders(Borders::ALL).title("input")),
                area,
            );
        }
    }
}

fn centered(area: Rect, w: u16, h: u16) -> Rect {
    let w = w.min(area.width);
    let h = h.min(area.height);
    Rect {
        x: area.x + (area.width.saturating_sub(w)) / 2,
        y: area.y + (area.height.saturating_sub(h)) / 2,
        width: w,
        height: h,
    }
}

fn format_bytes(n: u64) -> String {
    const K: f64 = 1024.0;
    let x = n as f64;
    if x >= K * K * K {
        format!("{:.1}G", x / (K * K * K))
    } else if x >= K * K {
        format!("{:.1}M", x / (K * K))
    } else if x >= K {
        format!("{:.1}K", x / K)
    } else {
        format!("{n}B")
    }
}

fn spawn_server(
    model: &Path,
    exe: Option<&Path>,
    backend: &str,
    threads: usize,
    ctx: usize,
    max_tokens: usize,
    tx: Sender<UiEvent>,
) -> io::Result<Child> {
    let port = free_port()?;
    let host = "127.0.0.1";
    let bin = exe
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("oxidize"));
    let mut cmd = Command::new(&bin);
    cmd.arg("serve")
        .arg(model)
        .arg("--host")
        .arg(host)
        .arg("--port")
        .arg(port.to_string())
        .arg("--backend")
        .arg(backend)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    if threads > 0 {
        cmd.arg("--threads").arg(threads.to_string());
    }
    if ctx > 0 {
        cmd.arg("--ctx-size").arg(ctx.to_string());
    }
    if max_tokens > 0 {
        cmd.arg("--max-tokens").arg(max_tokens.to_string());
    }
    let _ = tx.send(UiEvent::Log(format!(
        "$ {} serve {} --port {port}",
        bin.display(),
        model.display()
    )));
    let mut child = cmd.spawn()?;
    if let Some(out) = child.stdout.take() {
        pump(out, tx.clone());
    }
    if let Some(err) = child.stderr.take() {
        pump(err, tx.clone());
    }
    let url = format!("http://{host}:{port}");
    thread::spawn(move || wait_ready(url, tx));
    Ok(child)
}

fn pump<R: Read + Send + 'static>(r: R, tx: Sender<UiEvent>) {
    thread::spawn(move || {
        let reader = BufReader::new(r);
        for line in reader.lines().map_while(Result::ok) {
            if tx.send(UiEvent::Log(line)).is_err() {
                break;
            }
        }
    });
}

fn wait_ready(url: String, tx: Sender<UiEvent>) {
    for _ in 0..6000 {
        match probe(&url) {
            Ok(true) => {
                let id = list_model_id(&url).unwrap_or_else(|| "oxidize-default".into());
                let _ = tx.send(UiEvent::Ready { url, model_id: id });
                return;
            }
            Ok(false) => {}
            Err(_) => {}
        }
        thread::sleep(Duration::from_millis(300));
    }
    let _ = tx.send(UiEvent::Failed("server did not become ready".into()));
}

fn attach_worker(url: String, tx: Sender<UiEvent>) {
    for _ in 0..40 {
        if probe(&url).ok() == Some(true) {
            let id = list_model_id(&url).unwrap_or_else(|| "oxidize-default".into());
            let _ = tx.send(UiEvent::Ready {
                url,
                model_id: id,
            });
            return;
        }
        thread::sleep(Duration::from_millis(250));
    }
    let _ = tx.send(UiEvent::Failed(format!("no server at {url}")));
}

fn probe(url: &str) -> Result<bool, ureq::Error> {
    let resp = ureq::get(&format!("{url}/readyz"))
        .timeout(Duration::from_secs(2))
        .call()?;
    Ok(resp.status() == 200)
}

fn list_model_id(url: &str) -> Option<String> {
    let resp = ureq::get(&format!("{url}/v1/models"))
        .timeout(Duration::from_secs(3))
        .call()
        .ok()?;
    let v: Value = resp.into_json().ok()?;
    v.get("data")
        .and_then(|d| d.as_array())
        .and_then(|a| a.first())
        .and_then(|m| m.get("id"))
        .and_then(|id| id.as_str())
        .map(str::to_owned)
}

fn stream_chat(
    url: String,
    model: String,
    history: Vec<(String, String)>,
    params: (f32, f32, usize, usize),
    tx: Sender<UiEvent>,
) {
    let messages: Vec<Value> = history
        .into_iter()
        .map(|(role, content)| serde_json::json!({"role": role, "content": content}))
        .collect();
    let mut body = serde_json::json!({
        "model": model,
        "messages": messages,
        "stream": true,
        "temperature": params.0,
        "top_p": params.1,
        "max_tokens": params.3,
    });
    if params.2 > 0 {
        body["top_k"] = Value::from(params.2 as u64);
    }
    let result = ureq::post(&format!("{url}/v1/chat/completions"))
        .timeout(Duration::from_secs(600))
        .send_json(body);
    match result {
        Ok(resp) => {
            let mut reader = resp.into_reader();
            let mut buf = String::new();
            let mut tmp = [0u8; 2048];
            loop {
                match reader.read(&mut tmp) {
                    Ok(0) => break,
                    Ok(n) => {
                        buf.push_str(&String::from_utf8_lossy(&tmp[..n]));
                        while let Some(idx) = buf.find('\n') {
                            let line = buf[..idx].trim_end_matches('\r').to_string();
                            buf.drain(..=idx);
                            if let Some(delta) = sse_delta(&line) {
                                let _ = tx.send(UiEvent::Delta(delta));
                            }
                        }
                    }
                    Err(_) => break,
                }
            }
            let _ = tx.send(UiEvent::ChatDone);
        }
        Err(err) => {
            let _ = tx.send(UiEvent::ChatErr(err.to_string()));
        }
    }
}

fn sse_delta(line: &str) -> Option<String> {
    let line = line.trim();
    if !line.starts_with("data:") {
        return None;
    }
    let payload = line[5..].trim();
    if payload.is_empty() || payload == "[DONE]" {
        return None;
    }
    let v: Value = serde_json::from_str(payload).ok()?;
    v.get("choices")
        .and_then(|c| c.get(0))
        .and_then(|c| c.get("delta").and_then(|d| d.get("content")).or_else(|| c.get("text")))
        .and_then(|c| c.as_str())
        .map(str::to_owned)
}

fn parse_metrics(text: &str) -> Metrics {
    let mut flat: HashMap<String, f64> = HashMap::new();
    for raw in text.lines() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some(sp) = line.rfind(' ') else { continue };
        let key = &line[..sp];
        let Ok(value) = line[sp + 1..].parse::<f64>() else { continue };
        let name = key.split('{').next().unwrap_or(key);
        *flat.entry(name.to_string()).or_insert(0.0) += value;
    }
    let g = |k: &str| *flat.get(k).unwrap_or(&0.0);
    Metrics {
        tokens_per_second: g("oxidize_tokens_per_second"),
        tokens_generated: g("oxidize_tokens_generated_total"),
        requests_in_flight: g("oxidize_requests_in_flight"),
        requests_total: g("oxidize_requests_total"),
        queue_depth: g("oxidize_queue_depth"),
        kv_cache_bytes: g("oxidize_kv_cache_size_bytes"),
        errors_total: g("oxidize_errors_total"),
        error: None,
    }
}

fn start_metrics(url: String, tx: Sender<UiEvent>) {
    thread::spawn(move || loop {
        match ureq::get(&format!("{url}/metrics"))
            .timeout(Duration::from_secs(2))
            .call()
        {
            Ok(resp) => {
                if let Ok(text) = resp.into_string() {
                    let _ = tx.send(UiEvent::Metrics(parse_metrics(&text)));
                }
            }
            Err(err) => {
                let mut m = Metrics::default();
                m.error = Some(err.to_string());
                let _ = tx.send(UiEvent::Metrics(m));
            }
        }
        thread::sleep(Duration::from_secs(1));
    });
}

fn scan_models_async(tx: Sender<UiEvent>) {
    thread::spawn(move || {
        let rows = scan_models();
        let _ = tx.send(UiEvent::Models(rows));
    });
}

fn scan_models() -> Vec<ModelRow> {
    let mut roots = Vec::new();
    if let Ok(extra) = std::env::var("OXIDIZE_MODELS") {
        roots.extend(extra.split(':').filter(|s| !s.is_empty()).map(PathBuf::from));
    }
    if let Ok(cwd) = std::env::current_dir() {
        roots.push(cwd.join("models"));
    }
    if let Some(home) = std::env::var_os("HOME") {
        let home = PathBuf::from(home);
        roots.push(home.join(".cache/oxidize/hf"));
        roots.push(home.join(".cache/huggingface/hub"));
        roots.push(home.join("models"));
    }
    let mut out = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for root in roots {
        walk_gguf(&root, 4, 400, &mut out, &mut seen);
    }
    out.sort_by(|a, b| b.size.cmp(&a.size));
    out
}

fn walk_gguf(
    dir: &Path,
    depth: i32,
    budget: usize,
    out: &mut Vec<ModelRow>,
    seen: &mut std::collections::HashSet<PathBuf>,
) {
    if depth < 0 || out.len() >= budget {
        return;
    }
    let Ok(rd) = fs::read_dir(dir) else { return };
    for entry in rd.flatten() {
        if out.len() >= budget {
            return;
        }
        let path = entry.path();
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.starts_with('.') {
            continue;
        }
        if path.is_dir() {
            walk_gguf(&path, depth - 1, budget, out, seen);
        } else if name.to_ascii_lowercase().ends_with(".gguf") && seen.insert(path.clone()) {
            let size = entry.metadata().map(|m| m.len()).unwrap_or(0);
            let facts = peek_gguf(&path).unwrap_or_default();
            out.push(ModelRow {
                path,
                name,
                size,
                source: dir
                    .file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or("disk")
                    .to_string(),
                facts,
            });
        }
    }
}

fn peek_gguf(path: &Path) -> io::Result<String> {
    let mut f = fs::File::open(path)?;
    let mut magic = [0u8; 4];
    f.read_exact(&mut magic)?;
    if &magic != b"GGUF" {
        return Err(io::Error::other("not gguf"));
    }
    Ok("GGUF".into())
}

fn free_port() -> io::Result<u16> {
    let listener = TcpListener::bind("127.0.0.1:0")?;
    Ok(listener.local_addr()?.port())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn fuzzy_matches_case_insensitive() {
        assert!(fuzzy("chat", "Go to Chat"));
        assert!(!fuzzy("xyz", "Go to chat"));
        assert!(fuzzy("", "anything"));
    }

    #[test]
    fn sse_extracts_delta_content() {
        let line = r#"data: {"choices":[{"delta":{"content":"Hi"}}]}"#;
        assert_eq!(sse_delta(line).as_deref(), Some("Hi"));
        assert_eq!(sse_delta("data: [DONE]"), None);
        assert_eq!(sse_delta("event: ping"), None);
    }

    #[test]
    fn metrics_sum_unlabeled_series() {
        let text = "oxidize_tokens_per_second 12.5\noxidize_requests_total 3\n# comment\n";
        let m = parse_metrics(text);
        assert_eq!(m.tokens_per_second, 12.5);
        assert_eq!(m.requests_total, 3.0);
    }

    #[test]
    fn peek_rejects_non_gguf() {
        let dir = std::env::temp_dir().join(format!(
            "oxidize-tui-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("x.bin");
        fs::write(&path, b"nope").unwrap();
        assert!(peek_gguf(&path).is_err());
        let _ = fs::remove_dir_all(dir);
    }
}
