# Vendored Criterion

A vendored copy of the [Criterion](https://github.com/Snaipe/Criterion) C/C++
unit testing framework, used by `make test` to run the `tests/test_*.c` suite.

## License

Criterion is distributed under the **MIT License** (see `LICENSE` in this
directory). The MIT license is preserved verbatim, including the original
copyright notice:

> Copyright (c) 2015-2017 Franklin "Snaipe" Mathieu <http://snai.pe/>

## Provenance

| Field         | Value                                   |
|---------------|-----------------------------------------|
| Upstream      | https://github.com/Snaipe/Criterion     |
| Version       | 2.4.3 (release tag `v2.4.3`)            |
| License       | MIT                                     |
| Fetch method  | `git clone --depth 1 --branch v2.4.3`  |

## Layout

```
tests/criterion/
├── LICENSE                  # Verbatim upstream MIT license
├── README.md                # This file
├── include/
│   └── criterion/           # Public headers (use #include <criterion/criterion.h>)
│       ├── criterion.h      # Umbrella header
│       ├── assert.h         # cr_assert / cr_expect / cr_assert_eq / ...
│       ├── hooks.h
│       └── ...               # types.h, options.h, redirect.h, etc.
└── lib/
    └── libcriterion.a       # Combined static library
```

`libcriterion.a` is a **combined** archive produced by merging Criterion's own
`libcriterion.a` with its bundled subproject libraries (`libboxfort.a`,
`libnanomsg.a`, `libprotobuf_nanopb_static.a`). All objects are concatenated
into a single archive via `ar x` + `ar rcs` so that downstream linkers need only
one `-l` entry.

## Build flags used to produce `libcriterion.a`

```bash
meson setup -Ddefault_library=static \
            -Dtests=false \
            -Dsamples=false \
            -Ddev=false \
            -Dcxx-support=disabled \
            -Di18n=disabled \
            -Dtheories=disabled \
            -Ddiffs=disabled \
            build-static
ninja -C build-static
```

Disabled features rationale:
- `tests=false`, `samples=false` — we don't need upstream's self-tests.
- `cxx-support=disabled` — the C port is pure C11, no C++ tests.
- `i18n=disabled` — no localization needed.
- `theories=disabled` — requires libffi; not needed for our simple assertions.
- `diffs=disabled` — string diff rendering; not needed.

## Link-time requirements

`libcriterion.a` has the following system library dependencies that the
`Makefile` already passes on the `test_runner` link line:

- `-lpthread` — thread primitives used by the worker pool.
- `-lrt` — `shm_open`/`shm_unlink` for test isolation via shared memory.
- `-ldl` — `dlopen` for runtime plugin loading.
- `-lm` — math helpers.

The `Makefile`'s `TEST_LDFLAGS` already includes all four.

## Updating

To refresh the vendored copy (e.g. when a Criterion security advisory lands or
the mission bumps to Criterion 3.x):

```bash
# 1. Clone and build as above.
# 2. Recombine the static library:
mkdir combine && cd combine
ar x /path/to/build-static/src/libcriterion.a
ar x /path/to/build-static/subprojects/nanomsg/libnanomsg.a
ar x /path/to/build-static/subprojects/nanopb/libprotobuf_nanopb_static.a
# (boxfort objects are already bundled into libcriterion.a by meson)
ar rcs lib/libcriterion.a *.o
# 3. Replace include/ and lib/ in this directory.
# 4. Preserve LICENSE verbatim.
# 5. Run `make -C oxidize-c test` to verify all 8+ test suites still pass.
```
