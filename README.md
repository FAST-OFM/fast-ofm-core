<p align="center">
  <a href="https://github.com/FAST-OFM">
    <img src="https://github.com/FAST-OFM.png?size=200" alt="Fast OFM logo" width="132">
  </a>
</p>

<h1 align="center">Fast OFM Core</h1>

<p align="center">
  <strong>Independent RG focus, calibration, focus-surface and scan-planning research</strong>
</p>

<p align="center">
  <a href="https://github.com/FAST-OFM/fast-ofm">Project index</a> ·
  <a href="https://github.com/FAST-OFM/openflexure-wsi">GPL integration</a> ·
  <a href="https://github.com/FAST-OFM/fast-ofm-stitching-openflexure">LGPL stitching worker</a> ·
  <a href="https://github.com/FAST-OFM/controller">Controller</a> ·
  <a href="https://github.com/FAST-OFM/hardware">Hardware</a>
</p>

<p align="center">
  <sub>Research prototype · Source-available for noncommercial use · Not for diagnostic use</sub>
</p>

Private release-candidate workspace for Alexander Fridman's independently
authored Fast OFM research algorithms. This package is intentionally separate from the GPL
OpenFlexure Microscope Server and communicates through a versioned process
protocol rather than Python imports.

Status: pre-release, private, not validated for diagnostic or clinical use.

## Current capability

The private checkpoint provides a persistent JSON-lines process service for:

- deterministic serpentine planning, including disconnected regions;
- sparse focus-surface fitting and prediction;
- tissue-bound R/G shift measurement from verified frame artifacts;
- processed-JPEG flat-field fitting, application and independent holdout QC;
- WHITE tissue-field preparation, crop expansion and reusable focus windows;
- signed R/G calibration fitting from an immutable observation series;
- simultaneous RED+GREEN RAW spectral calibration, one-frame focus
  measurement, diagnostic peripheral recovery and signed curve evaluation;
- exact frozen-grid planning without copying the GPL OpenFlexure planner;
- validated stitching-job submission to a separately installed LGPL worker.

The stitching capability appears in `core.capabilities` only when the
replaceable `fast-ofm-stitch-openflexure` executable is installed. The core
verifies every source tile and records the requested worker/cache limits; it
does not import `openflexure-stitching`.

Every file artifact is constrained to an explicitly configured root and
verified by declared media type, byte size and SHA-256 before it is read. The
process has no camera, illumination, stage or OpenFlexure imports. The GPL
server retains those hardware responsibilities and exchanges only JSON plus
digest-verified file artifacts with this process.

```bash
python -m fast_ofm_core capabilities
python -m fast_ofm_core serve-jsonl --artifact-root /path/to/exchange
```

The service reads one request envelope per line from standard input and writes
one response envelope per line to standard output. Logs must go to standard
error so protocol output remains machine-readable.

See [PROTOCOL.md](PROTOCOL.md) for the complete versioning, artifact and error
contract, and [examples/protocol](examples/protocol) for deterministic synthetic
request/response fixtures that run without microscope access.

## Licensing

Original implementation is intended for release under PolyForm Noncommercial
1.0.0. Commercial use requires a separate written license. Neutral protocol
schemas are Apache-2.0 so independently licensed clients can interoperate. See
`LICENSING.md` and `AUTHORS.md`; third-party or LGPL-derived code is not
admitted to this tree.
