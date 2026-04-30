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
