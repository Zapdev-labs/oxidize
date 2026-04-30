use assert_cmd::Command;

#[test]
fn help_reports_llamas_cli_binary() {
    let mut cmd = Command::cargo_bin("llamas-cli").expect("binary should build");
    let assert = cmd.arg("--help").assert().success();
    let output = String::from_utf8(assert.get_output().stdout.clone()).expect("utf8");
    assert!(
        output.contains("llamas-cli"),
        "expected help output to contain binary name, got: {output}"
    );
}

#[test]
fn default_mode_runs_single_shot_inference() {
    let mut cmd = Command::cargo_bin("llamas-cli").expect("binary should build");
    let assert = cmd.arg("--prompt").arg("ping").assert().success();
    let output = String::from_utf8(assert.get_output().stdout.clone()).expect("utf8");
    assert!(output.contains("generation progress: 1/2 tokens"));
    assert!(output.contains("generation progress: 2/2 tokens"));
    assert!(output.contains("llamas-cli: ping"));
    assert!(output.contains("generation stats: tokens=2 speed="));
    assert!(output.contains(" tok/s"));
}
