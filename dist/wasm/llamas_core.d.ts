/* tslint:disable */
/* eslint-disable */

export interface LlamasWorkerModelConfig {
    vocab_size: number;
    context_size: number;
    layer_count: number;
}

export interface LlamasWorkerInferenceRequest {
    prompt_tokens: number[];
    max_new_tokens: number;
    model: LlamasWorkerModelConfig;
}

export interface LlamasWorkerInferenceResponse {
    generated_tokens: number[];
    consumed_tokens: number;
}

export interface LlamasWorkerStreamChunk {
    token: number;
    index: number;
}

export interface LlamasWorkerMessageResponse {
    response: LlamasWorkerInferenceResponse | null;
    error: string | null;
}

export interface LlamasWorkerStreamResponse {
    chunks: LlamasWorkerStreamChunk[];
    response: LlamasWorkerInferenceResponse | null;
    error: string | null;
}



export function wasm_collect_worker_stream(request_json: string): string;

export function wasm_handle_worker_message(request_json: string): string;

export function wasm_workspace_status(): string;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly wasm_collect_worker_stream: (a: number, b: number) => [number, number];
    readonly wasm_handle_worker_message: (a: number, b: number) => [number, number];
    readonly wasm_workspace_status: () => [number, number];
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
