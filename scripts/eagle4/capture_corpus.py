#!/usr/bin/env python3
"""Prefill many short documents on a qwen4exp GGUF and dump EAGLE-4 features.

One oxidize-c process per document. Context stays under the QSA budget
(2048). --n-predict 0 keeps the run to the prefill that fills the capture.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

DOCS = [
    "A hash table stores keys in an array of buckets. The hash function maps a key to an index. When two keys land in the same bucket, chaining keeps a short list there. A good table grows when the load factor passes about 0.7, rehashing every live key into a larger array. Lookup is expected constant time, and the worst case is a long chain.",
    "Binary search finds a value in a sorted array. Keep a low index and a high index. Compare the middle element to the target. If the middle is too small, move low to the middle plus one. If it is too large, move high to the middle minus one. Stop when low passes high or the middle matches. The number of steps is logarithmic in the length.",
    "Mergesort splits a list in half, sorts each half, and merges the two sorted halves. The merge walks two pointers and always takes the smaller head. The depth of the recursion is log n, and each level copies every element once, so the total work is n log n. The algorithm needs an extra buffer the size of the input.",
    "A ring buffer implements a queue in a fixed array. Head is the next item to read and tail is the next slot to write. Increment both with a modulo of the capacity. The queue is empty when head equals tail, and full when the next tail would equal head. One slot is left unused so those two states stay distinct.",
    "Breadth-first search explores a graph level by level. Put the start vertex in a queue and mark it seen. While the queue is not empty, pop a vertex and push each unseen neighbor, recording the parent that discovered it. The parent pointers reconstruct the shortest path in an unweighted graph. The seen set prevents cycles from looping forever.",
    "A recursive function calls itself on a smaller input. It needs a base case that returns immediately, otherwise the call stack grows without bound. Computing a factorial multiplies n by factorial of n minus one, and returns one when n is zero. Tail calls can be rewritten as a loop that updates an accumulator.",
    "Floating-point numbers keep a sign, an exponent, and a fraction. Adding a very small value to a very large value can round the small value away entirely. Compare floats with a tolerance, not with equality. Summing many tiny values is more accurate if you add them from smallest to largest, or with a compensated summation.",
    "A unit test calls one function with a fixed input and checks the output. Keep tests independent so the order does not matter. A failing assertion should name the input and the actual value. Run the suite on every change. Property tests generate many inputs and check a rule, such as encode then decode returning the original bytes.",
    "An allocator hands out blocks from a heap. A free list links unused blocks. Allocation pops a block, and free pushes it back. If the request is larger than any free block, the allocator asks the operating system for more pages. Fragmentation happens when free space is split into pieces too small to satisfy a request.",
    "A recursive-descent parser has one function per grammar rule. The function peeks at the next token and decides which alternative to take. On success it consumes the tokens it recognized and returns a tree node. On failure it reports the token it did not expect and the position. Precedence is encoded by the order of the rules, not by backtracking over the whole file.",
    "A cache keeps recent results so a repeated question is cheap. An LRU cache stores entries in a hash map and a doubly linked list. A hit moves the entry to the front. A miss computes the value, inserts it, and if the cache is over capacity, drops the tail. The map makes the move constant time.",
    "A mutex protects a shared variable. The thread that locks it is the only one allowed to read or write that variable. Unlocking lets another waiter in. Holding a lock across a slow call increases contention. Two locks acquired in opposite orders can deadlock, so every thread should take them in the same order.",
    "UTF-8 encodes a code point in one to four bytes. ASCII characters stay a single byte. A decoder rejects overlong encodings and sequences that do not start with a valid lead byte. String length in bytes is not the number of characters. Slicing should cut on character boundaries, not in the middle of a code point.",
    "Matrix multiplication of an m by k matrix and a k by n matrix produces an m by n matrix. The inner dimension must match. The naive algorithm is three nested loops. Tiling the loops so a small block stays in cache reduces memory traffic. The product is not commutative: A times B is a different shape and value from B times A when the matrices are not square.",
    "Functions should return an error instead of crashing when a file is missing. The caller decides whether to retry, skip, or stop. Error values should include the path and the operating system message. Ignoring an error hides the first failure and makes the later crash harder to understand. Propagate the error until a layer knows how to present it.",
    "A bloom filter answers whether a key might be in a set. It stores a bit array and several hash functions. Insert sets the bits those hashes select. A query checks those bits. If any bit is clear, the key is absent. If all are set, the key might be present, because different keys can share bits. The filter never forgets an inserted key and never stores the key itself.",
    "Dijkstra's algorithm finds shortest paths on a graph with non-negative weights. A priority queue holds vertices ordered by the best distance known so far. Pop the closest vertex and relax each outgoing edge. If a neighbor's distance improves, update its place in the queue. Once a vertex is popped, its distance is final.",
    "A writer for a length-prefixed format first writes a four-byte count, then the payload. The reader reads the count and then exactly that many bytes. If the stream ends early, the message is truncated and should be rejected. Lengths should be checked against a maximum so a corrupt count cannot force a huge allocation.",
    "Quicksort picks a pivot and partitions the array into values less than the pivot and values greater than it. It then sorts the two sides. Average time is n log n. A sorted input with a bad pivot choice degrades to n squared. Choosing the median of the first, middle, and last elements avoids that pattern. The sort is in place and not stable.",
    "A small interpreter walks an abstract syntax tree. A number node returns its value. A binary node evaluates the left side, evaluates the right side, and applies the operator. A variable node looks up the name in an environment map. A let node binds a name to a value in a child environment so the binding disappears when the node finishes.",
    "A byte buffer grows by doubling. Append copies the new bytes and, when size would pass capacity, allocates a new array twice as large and copies the old bytes over. Doubling makes the total copy cost linear in the final size, because each byte is copied once per growth and the sizes form a geometric series. Starting from a small capacity wastes a little memory and avoids a copy on every single append.",
    "Topological sort orders the vertices of a directed acyclic graph so every edge points forward. Compute the indegree of each vertex. Queue every vertex whose indegree is zero. Pop a vertex, emit it, and decrement the indegree of each successor. A successor that reaches zero joins the queue. If vertices remain at the end, the graph has a cycle and no valid order exists.",
    "A checksum detects accidental changes. Adler-32 keeps two running sums. One is the sum of the bytes and the other is the sum of those sums, both modulo a large prime. The two values are packed into a 32-bit word. It is fast and catches swapped blocks better than a plain sum. It is not a cryptographic hash: an attacker can craft a collision on purpose.",
    "Memory mapping a file lets the kernel page the bytes in as they are touched. The program sees a pointer and a length. Reading a page that is not resident faults, and the kernel fills it from disk. Writing a private mapping copies the page on the first store so other processes keep the original. Unmapping releases the address range. The file itself stays on disk.",
    "A state machine for a lexer starts in a ground state. Letters accumulate into an identifier. Digits accumulate into a number. A quote starts a string that ends at the next unescaped quote. Whitespace is skipped. Any other character is a one-character token. The machine returns the token kind and the slice of the source it consumed, then resumes at the following byte.",
    "Gradient descent updates a vector of parameters by subtracting a scaled gradient. The learning rate chooses the step size. Too large a rate oscillates and diverges. Too small a rate crawls. Momentum keeps a velocity that is a blend of the previous velocity and the new gradient, which smooths noise. Adam tracks a mean and a second moment and scales each coordinate by the root of that second moment.",
    "A priority queue can be a binary heap. The smallest element sits at index zero. The children of index i are at 2i+1 and 2i+2. Inserting writes the value at the end and swaps it upward while it is smaller than its parent. Popping moves the last value to the root and swaps it downward with the smaller child until the heap property holds. Both operations are logarithmic.",
    "Copy-on-write shares a buffer between readers. A reference count starts at one. A reader increments it. When a writer wants to change the bytes and the count is greater than one, it allocates a private copy and decrements the shared count. When the count hits zero the buffer is freed. Readers never see a partial write because the writer mutates only the private copy.",
    "A log-structured merge tree writes new keys into a memory table. When the table is full it is flushed to disk as a sorted run. A read checks the memory table first, then the runs from newest to oldest. Compaction merges several runs into one, dropping overwritten keys. The design turns random writes into sequential writes, at the cost of reading more than one run.",
    "Backpressure slows a producer when the consumer is behind. A bounded channel blocks or returns an error when it is full. The producer then stops reading from its own source, so the limit propagates upstream. Without a bound, a fast producer allocates without limit and the process runs out of memory. The bound should match the amount of work the consumer can finish in a short time.",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--oxidize", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ctx", type=int, default=1536)
    ap.add_argument("--layers", default="2,24,46")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    failed = 0
    for i, text in enumerate(DOCS):
        src = args.out / f"doc-{i:02d}.txt"
        dump = args.out / f"doc-{i:02d}.e4h1"
        src.write_text(text + "\n")
        if dump.exists() and dump.stat().st_size > 64:
            print(f"skip {dump.name}", flush=True)
            continue
        cmd = [
            str(args.oxidize),
            "--model", str(args.model),
            "--prompt-file", str(src),
            "--n-predict", "0",
            "--ctx", str(args.ctx),
            "--threads", "4",
            "--temperature", "0",
            "--repeat-penalty", "1",
            "--stream-experts",
            "--eagle-dump", str(dump),
            "--eagle-layers", args.layers,
        ]
        print(f"capture {i:02d}", flush=True)
        proc = subprocess.run(cmd, capture_output=True, text=True)
        log = args.out / f"doc-{i:02d}.log"
        log.write_text(proc.stderr + "\n" + proc.stdout)
        if proc.returncode != 0 or not dump.exists():
            failed += 1
            print(f"fail {i:02d} rc={proc.returncode}", flush=True)
        else:
            for line in proc.stderr.splitlines():
                if "eagle: wrote" in line or "prefill:" in line:
                    print(line, flush=True)
    if failed:
        print(f"{failed} documents failed", file=sys.stderr)
        raise SystemExit(1)
    print(f"captured {len(DOCS)} documents", flush=True)


if __name__ == "__main__":
    main()
