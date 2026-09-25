#!/usr/bin/env python3
"""Tokenizer parity harness: oxidize-c vs. a llama.cpp golden corpus.

Both drivers speak the same protocol: one hex-encoded UTF-8 text per stdin
line, space-separated ids per stdout line.

  * oxidize-c side:  ./oc-tokenize MODEL.gguf      (make oc-tokenize)
  * reference side:  llama-tokenize-ref MODEL.gguf (scripts/llama_tokenize_ref.cpp)

Commands:
  build  — tokenize the built-in corpus (+ seeded fuzz strings) with the
           reference driver and write a golden JSON.
  check  — tokenize every golden text with oc-tokenize and require exact
           id equality. Exit status 1 on any mismatch.

Golden JSON formats accepted by `check`:
  {"text": [ids], ...}                              (plain mapping)
  {"entries": [{"text"|"text_hex": ..., "ids": [...]}, ...]}
  {"cases": [{"text": ..., "ids_parse_special": [...],
              "ids_no_parse_special": [...]}, ...]}   (llama-server corpus)
`build` writes the second form (text_hex covers invalid UTF-8 inputs).

Examples:
  scripts/tokenizer_parity.py build --ref ./llama-tokenize-ref \\
      --model base-IQ3_M.gguf --out golden.json
  scripts/tokenizer_parity.py check --tool ./oc-tokenize \\
      --model base-IQ3_M.gguf --golden golden.json
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time


def base_corpus() -> list[bytes]:
    t = [
        "The capital of France is",
        "Hello world",
        " Hello world",
        "  Hello  world  ",
        "hello\nworld\n\n\nfoo\r\nbar\r\rbaz",
        "\n", " ", "  ", "\t", " \n ", "\n \n", "a  \n  b", "x   ", "   y",
        "I'm, you're, we've, they'll, he'd, it's, can't",
        "I'M YOU'RE WE'VE THEY'LL HE'D IT'S CAN'T",
        "don'T 'S 'Re 'vE 'lL 'd ' s '", "'ſ 'ſs 'x", "''s '''",
        "1", "12", "123", "1234", "12345", "1234567890", "3.14159 2,000,000 1e-10",
        "Question: What is 12345 + 6789?\nAnswer: 12345 + 6789 =",
        "٣٤٥٦ ১২৩৪ ①②③ ½¾ ⅷ",
        "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
        "for (int i = 0; i < n; ++i) { x[i] += y[i] * 2; }\n\treturn x;",
        "if __name__ == '__main__':\n    main()\n",
        "#include <stdio.h>\nint main(void){printf(\"hi\\n\");}",
        "<html><body class=\"x\">&amp;&lt;</body></html>",
        "东京是日本的首都，也是世界上人口最多的城市之一。",
        "日本語のテキスト、カタカナとひらがな。",
        "한국어 텍스트입니다.",
        "Привет, мир! Ёжик в тумане.",
        "Γειά σου Κόσμε",
        "مرحبا بالعالم",
        "שלום עולם",
        "नमस्ते दुनिया। क्षत्रिय",
        "ক্ষ্মী த்தமிழ் ಕನ್ನಡ",
        "e\u0301 a\u0300\u0301 n\u0303 \u0301x",
        "zero\u200bwidth \u200cjoin\u200d here",
        "क्\u200dष र\u200c्",
        "emoji 😀🎉 👨‍👩‍👧‍👦 🏳️‍🌈 👍🏽",
        "¡Hola! ¿Qué tal? «guillemets» „quotes“ “smart” ‘single’",
        "— – … • · ° ± × ÷ € £ ¥ © ® ™",
        "$100 + 50% = ^_^ ~~ || <=> `tick` @user #tag",
        "!!! ??? ... ,,, ;;; ::: \"\" ''",
        " !\n ?\r\n .\n\n",
        "a\u00a0b\u2003c\u3000d\u2028e\u0085f",
        "tab\tsep\tvalues\t\t\n",
        "Mixed123abc 456def78 9ghi",
        "x1y22z333w4444",
        "ALLCAPS lowercase CamelCase snake_case kebab-case",
        "https://example.com/path?q=1&r=2#frag",
        "user@example.com",
        "C:\\Windows\\System32",
        "<|ifm|im_start|>user\nWhat is the capital of France? Answer briefly.<|ifm|im_end|><|ifm|im_start|>assistant\n<ifm|think>\n",
        "<|ifm|begin_of_text|>hello<|ifm|endoftext|>",
        "text<ifm|think>\nreasoning\n</ifm|think>\nanswer<|ifm|im_end|>",
        "<|im_start|>user<|im_end|><|eot_id|><|endoftext|><|im_end|>",
        "<image> <video></video> <tool_response>x</tool_response>",
        "<|ifm|im_start|><|ifm|im_start|>|>|><<|ifm|im_end|",
        "almost <|ifm|im_start| special",
        "\u00ff\u0100\uffff\U0001f600\U0010fffd",
    ]
    # Digit-run prefixes (\p{N}{1,3} grouping) and the phase-0 golden corpus.
    t += ["1234567890123"[:n] for n in range(1, 14)]
    t += [
        "Hello,world!", "they'll we've I'm you'd she'S", "'s 't 're 've 'm 'll 'd",
        "trailing spaces   ", "multiple     spaces   here", "\n\n\n",
        "我喜欢编程。你呢？", "👨‍👩‍👧‍👦 family", "🇺🇸🇯🇵 flags",
        "café naïve résumé Ångström", "Straße über Größe", "ñandú pingüino",
        "العربية 123 نص", "x = {'a': [1, 2.5, -3e10]}", "$1,234,567.89 and 50% off",
        "12345+6789=19134", "2026-09-25T04:41:00Z", "((()))[[{}]]<>", "...!!!???---___",
        "a.b,c;d:e!f?g", "Ⅻ ℃ ™ © ® ° ± × ÷", " nbsp emspace​zwsp",
        "The quick brown fox jumps over the lazy dog.", " Paris", "Paris",
        " the United Kingdom",
    ]
    seen = set()
    t = [s for s in t if not (s in seen or seen.add(s))]
    out = [s.encode() for s in t]
    out.append("Lorem ipsum dolor sit amet, consectetur adipiscing elit. ".encode() * 40)
    out.append(("The quick brown fox jumps over the lazy dog 123. " * 400
                + "\n\n    indented code();\n" * 50).encode())
    out.append(("  " * 300 + "\n" * 50 + "\t" * 20 + "end").encode())
    out.append(b"invalid \xff\xfe utf8 \xc3( \xe2\x82 \xf0\x9f\x98 tail")
    out.append(b"\xc0\x80 overlong \xed\xa0\x80 surrogate")
    return out


FUZZ_ALPHABET = (
    list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
    + list(" \t\n\r'\"!?.,;:-_()[]{}<>|/\\@#$%^&*+=~`")
    + [" ", " ", " ", "\n", "'s", "'LL", "'", "ſ"]
    + list("éàüñçøßĳœÆ") + list("中文字日本語한국") + list("привет") + list("ابت")
    + ["\u0301", "\u0308", "\u093f", "\u094d", "\u200c", "\u200d", "\u00a0", "\u3000",
       "\u2028", "\u0085", "\u200b", "\ufeff"]
    + list("٣١২৩①½") + list("😀👍🏽🎉") + list("—…•€©™«»")
    + ["<|ifm|im_start|>", "<|ifm|im_end|>", "<ifm|think>", "</ifm|think>", "<|eot_id|>",
       "<|im_end|>", "<image>"]
)


def fuzz_corpus(n: int, seed: int) -> list[bytes]:
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        k = rng.randint(1, 60)
        out.append("".join(rng.choice(FUZZ_ALPHABET) for _ in range(k)).encode())
    return out


def run_driver(cmd: list[str], texts: list[bytes]) -> tuple[list[list[int]], float]:
    payload = "".join(t.hex() + "\n" for t in texts).encode()
    t0 = time.time()
    res = subprocess.run(cmd, input=payload, capture_output=True, check=False)
    dt = time.time() - t0
    if res.returncode != 0:
        sys.stderr.write(res.stderr.decode(errors="replace"))
        raise SystemExit(f"driver failed: {' '.join(cmd)} (exit {res.returncode})")
    sys.stderr.write(res.stderr.decode(errors="replace"))
    lines = res.stdout.decode().split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    if len(lines) != len(texts):
        raise SystemExit(f"driver returned {len(lines)} lines for {len(texts)} texts")
    out = []
    for ln in lines:
        if ln.startswith("ERROR"):
            out.append(None)
        else:
            out.append([int(x) for x in ln.split()] if ln.strip() else [])
    return out, dt


def load_golden(path: str, no_special: bool = False) -> list[tuple[bytes, list[int]]]:
    with open(path) as f:
        g = json.load(f)
    if isinstance(g, dict) and "cases" in g:
        # llama-server corpus (golden_gen.py): both parse_special variants.
        key = "ids_no_parse_special" if no_special else "ids_parse_special"
        return [(c["text"].encode(), c[key]) for c in g["cases"]]
    if isinstance(g, dict) and "entries" in g:
        items = []
        for e in g["entries"]:
            text = bytes.fromhex(e["text_hex"]) if "text_hex" in e else e["text"].encode()
            items.append((text, e["ids"]))
        return items
    if isinstance(g, dict):
        return [(k.encode(), v) for k, v in g.items()]
    if isinstance(g, list):
        return [(e["text"].encode() if "text" in e else bytes.fromhex(e["text_hex"]),
                 e["ids"]) for e in g]
    raise SystemExit("unrecognized golden format")


def flags(args) -> list[str]:
    f = []
    if args.no_special:
        f.append("--no-special")
    if args.bos:
        f.append("--bos")
    return f


def cmd_build(args) -> None:
    texts = base_corpus() + fuzz_corpus(args.fuzz, args.seed)
    ids, dt = run_driver([args.ref, args.model] + flags(args), texts)
    entries = []
    for t, i in zip(texts, ids):
        try:
            entries.append({"text": t.decode("utf-8"), "ids": i})
        except UnicodeDecodeError:
            entries.append({"text_hex": t.hex(), "ids": i})
    with open(args.out, "w") as f:
        json.dump({"source": "llama.cpp-k2 llama_tokenize (vocab_only)",
                   "parse_special": not args.no_special, "add_special": args.bos,
                   "entries": entries}, f, ensure_ascii=False, indent=1)
    print(f"wrote {args.out}: {len(entries)} texts ({dt:.2f}s)")


def cmd_check(args) -> None:
    items = load_golden(args.golden, args.no_special)
    texts = [t for t, _ in items]
    got, dt = run_driver([args.tool, args.model] + flags(args), texts)
    bad = 0
    for (text, want), have in zip(items, got):
        if have != want:
            bad += 1
            if bad <= args.show:
                print(f"MISMATCH {text[:80]!r}")
                n = 0
                while have and n < min(len(have), len(want)) and have[n] == want[n]:
                    n += 1
                print(f"  first diff at {n}: want {want[n:n + 8]} have "
                      f"{(have or [])[n:n + 8]} (len want {len(want)} have "
                      f"{len(have) if have is not None else 'ERR'})")
    total = len(items)
    print(f"{total - bad}/{total} exact matches ({100.0 * (total - bad) / max(total, 1):.2f}%), "
          f"{dt:.2f}s wall")
    sys.exit(1 if bad else 0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--ref", required=True)
    b.add_argument("--model", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--fuzz", type=int, default=400)
    b.add_argument("--seed", type=int, default=1234)
    c = sub.add_parser("check")
    c.add_argument("--tool", required=True)
    c.add_argument("--model", required=True)
    c.add_argument("--golden", required=True)
    c.add_argument("--show", type=int, default=10)
    for p in (b, c):
        p.add_argument("--no-special", action="store_true")
        p.add_argument("--bos", action="store_true")
    args = ap.parse_args()
    cmd_build(args) if args.cmd == "build" else cmd_check(args)


if __name__ == "__main__":
    main()
