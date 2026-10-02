//! server-001: startup posture when API auth is disabled.
//!
//! Runs the real `oxidize-server` binary (no model) and checks that a
//! non-loopback bind warns, that `--require-auth-on-public-bind` exits 2, and
//! that loopback binds never warn.

use std::io::Read;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

const WARNING: &str = "API auth is disabled";

fn server() -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_oxidize-server"));
    command
        .env_remove("OXIDIZE_API_KEY")
        .env_remove("OXIDIZE_API_KEYS")
        .env_remove("RUST_LOG")
        .args(["--port", "0"])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    command
}

/// Run until exit or `timeout`, then kill. Returns (exit code if it exited, stdout+stderr).
fn run_for(mut command: Command, timeout: Duration) -> (Option<i32>, String) {
    let mut child = command.spawn().expect("spawn oxidize-server");
    let deadline = Instant::now() + timeout;
    let code = loop {
        if let Some(status) = child.try_wait().expect("try_wait") {
            break status.code();
        }
        if Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            break None;
        }
        std::thread::sleep(Duration::from_millis(50));
    };
    let mut output = String::new();
    child
        .stdout
        .take()
        .unwrap()
        .read_to_string(&mut output)
        .unwrap();
    child
        .stderr
        .take()
        .unwrap()
        .read_to_string(&mut output)
        .unwrap();
    (code, output)
}

#[test]
fn public_bind_without_auth_warns_and_keeps_serving() {
    let mut command = server();
    command.args(["--host", "0.0.0.0"]);
    let (code, output) = run_for(command, Duration::from_secs(2));
    assert_eq!(code, None, "server should keep running; output:\n{output}");
    assert!(
        output.contains(WARNING),
        "missing warning; output:\n{output}"
    );
}

#[test]
fn public_bind_without_auth_exits_2_when_required() {
    let mut command = server();
    command.args(["--host", "0.0.0.0", "--require-auth-on-public-bind"]);
    let (code, output) = run_for(command, Duration::from_secs(10));
    assert_eq!(code, Some(2), "output:\n{output}");
    assert!(output.contains(WARNING), "output:\n{output}");
}

#[test]
fn loopback_bind_never_warns() {
    let mut command = server();
    command.args(["--host", "127.0.0.1", "--require-auth-on-public-bind"]);
    let (code, output) = run_for(command, Duration::from_secs(2));
    assert_eq!(code, None, "server should keep running; output:\n{output}");
    assert!(
        !output.contains(WARNING),
        "unexpected warning; output:\n{output}"
    );
}

#[test]
fn public_bind_with_auth_never_warns() {
    let mut command = server();
    command.env("OXIDIZE_API_KEY", "secret").args([
        "--host",
        "0.0.0.0",
        "--require-auth-on-public-bind",
    ]);
    let (code, output) = run_for(command, Duration::from_secs(2));
    assert_eq!(code, None, "server should keep running; output:\n{output}");
    assert!(
        !output.contains(WARNING),
        "unexpected warning; output:\n{output}"
    );
}
