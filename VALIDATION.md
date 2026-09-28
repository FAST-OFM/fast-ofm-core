# Validation

Recorded: 2026-09-28

This is a private prototype checkpoint. Passing software tests does not make
the package clinically validated or authorize microscope motion.

## Source validation

- host: private Linux x86_64 server;
- interpreter: CPython 3.11.9;
- optional native backend absent: `397 passed, 55 skipped` in 7.90 seconds;
- optional native backend installed from the x86_64 wheel: `452 passed` in
  4.13 seconds;
- the checked-in x86 CI command sequence independently completed REUSE, Ruff,
  required native-extension loading, `452 passed in 5.19s`, sdist and wheel
  builds in a fresh CPython 3.11 environment;
- the 55-test matrix covers native/portable NMI parity and runs when the
  extension is present;
- Ruff: clean for the complete core and GPL integration trees.

## Linux x86_64 clean installation

- host: private Fast OFM server;
- interpreter: fresh CPython 3.11.9 virtual environment;
- limits: 3 CPUs and 4 GiB RAM;
- install: fresh dependency resolution and non-editable wheel installation;
- optional native backend: built and imported as
  `_rg_nmi.cpython-311-x86_64-linux-gnu.so`;
- final result: `452 passed in 4.13s` from `site-packages`, with pytest's
  source-tree path configuration explicitly disabled;
- resource limits: 3 CPUs and 4 GiB RAM;
- native module: `_rg_nmi.cpython-311-x86_64-linux-gnu.so`.

## Linux ARM64 clean installation

- host: private Fast OFM server using Docker/QEMU ARM64 emulation;
- image: the same official multi-architecture `python:3.11-bookworm` digest;
- limits: 3 CPUs and 4 GiB RAM;
- install: fresh non-editable package installation inside the pinned image;
- optional native backend: built and imported as
  `_rg_nmi.cpython-311-aarch64-linux-gnu.so`;
- final result: `452 passed in 64.65s` under QEMU;
- resource limits: 3 CPUs and 4 GiB RAM;
- native module: `_rg_nmi.cpython-311-aarch64-linux-gnu.so`.

## Architecture and protocol gates

- core source has no imports of `openflexure_microscope_server`,
  `openflexure_stitching` or `labthings_fastapi`;
- all protocol schemas pass Draft 2020-12 schema validation;
- reference capability, route and focus-surface payloads validate;
- `rg.measure` verifies allow-listed NPZ/JSON artifacts by media type, size and
  SHA-256 before computing a tissue-bound shift;
- `rg.flat_field.*` and `rg.tissue_field.prepare` own the flat-field and WHITE
  tissue-mask mathematics outside the GPL process;
- `rg.simultaneous.*` owns RAW spectral calibration, unmixing, NMI focus
  measurement, peripheral diagnostics and signed focus-curve evaluation;
- `calibration.fit` verifies an immutable observation series and returns a
  reloadable candidate or valid signed model;
- `calibration.validate` recomputes persisted empirical profile evidence and
  the live residual focus budget before the GPL workflow may use it;
- calibration fitting, signed projection, cross-track evaluation, activation
  gate replay and empirical profile validation are absent from the shipped GPL
  model module; its remaining classes are process-boundary data contracts;
- GPL adapter invokes the core as a separate persistent process, preflights the
  prediction capability and validates every numeric field before using a Z target;
- sparse-focus plane/row mathematics is absent from the GPL scheduler; only
  measured-anchor state, hardware cadence and WHITE fallback policy remain there;
- signed correction, cross-track gating and empirical-error budgeting are absent
  from the GPL decision module and are enforced by `rg.decide` in the core;
- exact signed rectangular routes cross the process boundary and round-trip to
  the original frozen stage coordinates;
- `stitching.run` verifies a SHA-256 tile manifest and invokes only the
  separately installed LGPL worker executable;
- the worker receives the exact verified manifest and exposes only those tiles
  to upstream folder discovery, so a prior OME-TIFF cannot be re-read as an
  input frame.
- transient stitching failures preserve their protocol-level `retryable`
  classification.
- persistent JSON-lines median handshake latency was 0.744 ms on the x86
  development host, versus 1067.903 ms for one new process per request.

## Standalone 91-tile stitching replay

- host: private Linux x86_64 server while unrelated services remained running;
- input: 91 retained 4056×3040 JPEG fields with the existing full-correlation
  cache (158 loaded pairs, 157 accepted for final optimisation);
- settings: 3 workers, 4 GiB cache, full correlation, resize 0.2, overlap 0.14;
- output: 33104×29126 RGB uint8 OME-BigTIFF, 2,552,787,826 bytes;
- pyramid: 8 levels from 33104×29126 down to 258×227;
- worker time: 242.75 s on host libvips 8.9; end-to-end shell wall time:
  249.44 s; measured peak RSS: 3,990,072 KiB; no swap;
- exact parity against the retained reference level 0: maximum absolute pixel
  difference 0 and mean absolute difference 0;
- compatibility finding: this libvips 8.9 multi-page file is useful for timing
  and pixel-parity checks but QuPath 0.5.1 exposes it as one level. Release OME
  output now requires libvips 8.10+ SubIFDs; incompatible requests fail before
  rendering, while DZI-only output remains available.

## GPL adapter end-to-end replay

- host: private Linux x86_64 server in a separate validation environment;
- path: OpenFlexure `FinalStitcher` → `fast-ofm-core run-request` → separately
  installed `fast-ofm-stitch-openflexure`;
- input: two retained 4056×3040 JPEG fields from the 91-tile scan;
- settings: 3 workers, 4 GiB cache, full correlation, resize 0.2;
- result: completed OME-BigTIFF and complete DZI pyramid;
- OME-BigTIFF: 81,235,878 bytes, SHA-256
  `12158bd589fcdfdf54d538c86fc5e7b87b5aa3ef48ac84d403ddc04b58f82760`;
- worker time: 8.53 s; adapter wall time: 10.11 s; measured peak RSS:
  380,576 KiB; no production services were stopped;
- cancellation uses a dedicated process session so cancelling the GPL action
  kills both the core wrapper and its resource-heavy worker.

## Non-editable wheel installation gate

- built three independent wheels on Linux x86_64: GPL server, noncommercial
  core with native C++ extension, and LGPL worker;
- installed all three into a fresh CPython 3.11 virtual environment; metadata
  confirmed no editable/direct-source installation;
- `core.capabilities` discovered the native NMI backend and separately
  installed stitching worker;
- a current two-field end-to-end replay ran entirely from `site-packages` under
  3 CPU / 4 GiB limits and completed in 5.89 s wall time with 365,392 KiB peak
  RSS and no swap;
- the replay directory deliberately contained an earlier 64.8 MB OME-TIFF;
  exact manifest isolation prevented it from entering source discovery;
- output: 4068×5624 RGB uint8 OME-BigTIFF, 64,782,770 bytes, one base IFD plus
  four SubIFD pyramid levels, and a complete DZI pyramid;
- level-0 comparison with the editable-install replay was exact
  (`max_abs=0`, `mean_abs=0`); byte size differs because the fresh environment
  uses the required libvips 8.10+ SubIFD representation instead of the
  historical libvips 8.9 multi-page benchmark representation;
- packaged GPL wheel contains the three adapter files and none of the removed
  `pyramidal_stitch.py`, `stage_stitch.py` or `stitch_acceleration.py` modules.
- packaged GPL client reports an absent executable as retryable
  `CORE_UNAVAILABLE` and rejects a mock protocol-2.0 service with
  `CORE_PROTOCOL_ERROR`.

## GPL integration regression

- private Linux x86_64 server, current extraction snapshot mounted read-only;
- clean CPython 3.13 container pinned to three CPU cores and 4 GiB RAM;
- standalone GPL adapter, with core absent: `1398 passed, 4 skipped, 3 xfailed`
  in 184.24 s; the four skips are ordinary test-level skips and the optional
  cross-package modules are excluded explicitly at collection time;
- full adapter with this core installed as a separate package: `1705 passed,
  3 xfailed` in 277.17 s;
- all three xfails are predeclared expected failures.

## Pending release gates

- run a bounded smoke test on native Raspberry Pi 4/5 hardware;
- owner provenance and patent/publication decisions;
- clean one-commit private release candidate.
