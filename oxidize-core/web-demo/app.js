import init, {
  wasm_collect_worker_stream,
  wasm_workspace_status,
} from "../../dist/wasm/oxidize_core.js";

const output = document.getElementById("output");

async function main() {
  await init();
  const status = wasm_workspace_status();
  output.textContent = JSON.stringify(status, null, 2);
  void wasm_collect_worker_stream;
}

main().catch((error) => {
  output.textContent = String(error);
});
