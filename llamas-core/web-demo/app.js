import init, { wasm_collect_worker_stream, wasm_workspace_status } from "../../dist/wasm/llamas_core.js";

const output = document.getElementById("output");
const promptInput = document.getElementById("prompt");
const runButton = document.getElementById("run-button");

function parsePromptTokens(value) {
  return value
    .split(",")
    .map((token) => token.trim())
    .filter((token) => token.length > 0)
    .map((token) => Number.parseInt(token, 10))
    .filter((token) => Number.isInteger(token));
}

function render(value) {
  output.textContent = value;
}

async function run() {
  render("Loading WASM...");
  await init();
  render(`WASM ready (${wasm_workspace_status()})`);

  runButton.addEventListener("click", () => {
    const promptTokens = parsePromptTokens(promptInput.value);
    if (promptTokens.length === 0) {
      render("Please provide at least one integer prompt token.");
      return;
    }

    const request = {
      prompt_tokens: promptTokens,
      max_new_tokens: 4,
      model: {
        vocab_size: 32000,
        context_size: 4096,
        layer_count: 32,
      },
    };
    const response = JSON.parse(wasm_collect_worker_stream(JSON.stringify(request)));
    render(JSON.stringify(response, null, 2));
  });
}

run().catch((error) => {
  render(`Failed to initialize demo: ${error}`);
});
