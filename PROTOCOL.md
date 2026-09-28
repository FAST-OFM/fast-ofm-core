# Fast OFM Core protocol v1 draft

Status: `IMPLEMENTED PRIVATE CHECKPOINT — additive capabilities may follow`

## Purpose

This protocol is the licensing and deployment boundary between the GPL-covered
OpenFlexure integration and the separately distributed Fast OFM Core. It is not
a Python API. Neither side exchanges Python classes, callbacks, shared-memory
object graphs or OpenFlexure-internal identifiers.

The same protocol must support a generic command-line client, an OpenFlexure
adapter and offline replay without changing the core implementation.

## Reference fixtures

The release repository includes deterministic, synthetic request/response
pairs under `examples/protocol/v1`. They cover an exact serpentine route and a
bounded focus-surface prediction, contain no specimen or machine identifiers,
and are replayed against the real dispatcher by
`tests/test_protocol_examples.py` on x86_64 and ARM64.

## Transport

The checkpoint transport is one persistent JSON-lines subprocess. The GPL
adapter starts the configured executable without a shell, keeps it warm across
coarse operations and serializes one request per line. This keeps the two
programs independently installable while avoiding an unnecessary network
listener. A Unix-domain socket or loopback HTTP transport may be added later
without changing the envelopes.

The selection is measured on the x86 development host with the installed
`0.1.0.dev0` scaffold: 100 requests over one warm JSON-lines process had a
0.744 ms median and 11.613 ms p95; starting a new process for each of 20
capability calls had a 1067.903 ms median and 2204.109 ms p95. The first warm
process request includes interpreter startup and reached 1583.487 ms. These are
transport microbenchmarks, not acquisition-performance claims.

Rules:

- the service does not listen on a non-loopback interface by default;
- request and response bodies are UTF-8 JSON;
- standard output contains protocol lines only and logs use standard error;
- the adapter sends no shell command string and validates response identity;
- large images and mosaics are referenced as immutable ordinary files;
- every input artifact includes a SHA-256 digest;
- the service reads only from configured input roots and writes only to a
  configured output root;
- output files are written atomically and returned with a digest;
- paths, digests and generic physical coordinates cross the boundary;
  OpenFlexure objects do not.

## Versioning

- protocol version: `1.0`;
- clients must call `core.capabilities` before using optional operations;
- unknown major versions are refused;
- additive optional fields are allowed only within the same major version;
- unknown fields are rejected by default to catch integration drift;
- calibration/profile schemas carry their own independent version.

## Request envelope

Every request contains:

- `protocol_version` — exactly `1.0` for this specification;
- `request_id` — caller-generated UUID;
- `operation` — one operation identifier;
- `timeout_ms` — caller budget, not a promise of completion;
- `payload` — operation-specific object.

The server must echo `request_id`. A request is completed, refused or failed in
the implemented `1.0` checkpoint. The envelope reserves `accepted` for a later
asynchronous capability; clients must not assume it is available unless
`core.capabilities` advertises the corresponding job operations. Retrying a
future mutating operation with the same `request_id` must not create a second
job.

## Response envelope

`status` is one of:

- `completed` — result is final;
- `accepted` — reserved for an advertised future asynchronous capability;
- `refused` — valid request but quality/safety policy rejected the operation;
- `failed` — processing or protocol failure;
- `cancelled` — cancellation reached a safe boundary.

Errors contain a stable code, human-readable message, retryability and optional
structured detail. Stack traces and host paths are not returned by default.

## Operations

| Operation | Mode | Purpose |
| --- | --- | --- |
| `core.capabilities` | synchronous | Report protocol, algorithm and optional native-backend capabilities |
| `rg.measure` | synchronous | Measure R/G displacement and propose bounded Z correction from immutable frame artifacts |
| `rg.decide` | synchronous | Apply a validated model and correction policy to one serialized R/G measurement without commanding hardware |
| `rg.flat_field.fit` | synchronous | Fit processed-JPEG flat fields from phase-matched source captures |
| `rg.flat_field.apply` | synchronous | Apply a verified flat field to one bounded R/G pair |
| `rg.flat_field.validate` | synchronous | Evaluate an independent flat-field holdout |
| `rg.tissue_field.prepare` | synchronous | Build a WHITE-derived tissue support field, reusable windows and diagnostic overlay |
| `rg.simultaneous.calibrate` | synchronous | Fit and validate RED/GREEN spectral response maps from RAW artifacts |
| `rg.simultaneous.focus.sample` | synchronous | Unmix one mixed RAW frame, measure focus shift and optionally inspect peripheral tissue |
| `rg.simultaneous.focus.plan` | synchronous | Produce the deterministic signed Z calibration order |
| `rg.simultaneous.focus.fit` | synchronous | Fit a signed 2D shift-to-defocus curve from fixed-grid observations |
| `rg.simultaneous.focus.evaluate` | synchronous | Validate optional peripheral evidence and project an accepted shift to defocus |
| `calibration.fit` | synchronous | Fit and validate a calibration bundle from a manifest of observations |
| `calibration.validate` | synchronous | Revalidate a persisted RG profile and optional live focus-error budget |
| `focus.surface.fit` | synchronous | Fit bounded surface state from trusted anchors |
| `focus.surface.predict` | synchronous | Predict or refuse Z at requested XY positions |
| `planning.route` | synchronous | Produce a bounded route from geometry and optional masks/polygons |
| `stitching.run` | synchronous | Build registered pyramidal OME-BigTIFF from a tile manifest; advertised only when the replaceable LGPL worker is installed |
| `job.status` | reserved, not advertised | Future progress/result query for an asynchronous job |
| `job.cancel` | reserved, not advertised | Future cancellation request at a documented safe boundary |

## Artifact references

An artifact reference contains:

- `uri` — initially a `file:` URI under an allowed root;
- `sha256` — digest of exact bytes;
- `media_type` — for example `image/jpeg`, `image/tiff` or
  `application/json`;
- optional byte length and logical role.

The core verifies digest and root containment before decoding. Inputs are never
modified in place. A request may reference a manifest rather than list thousands
of tile paths in the envelope.

The implemented R/G boundary uses
`application/vnd.fast-ofm.rg-frame+npz` for bounded arrays and frame metadata,
`application/vnd.fast-ofm.rg-calibration+json` for the immutable measurement
policy/profile bundle, and
`application/vnd.fast-ofm.rg-calibration-series+json` for observations sent to
`calibration.fit`. Flat-field, tissue-field and simultaneous-RAW operations use
their own bounded NPZ media types and exact member sets; results are returned
as new digest-verified artifacts rather than mutating inputs.

## Coordinate conventions

- physical XY and Z values use micrometres;
- image dimensions use integer pixels;
- pixel sizes use micrometres per pixel;
- route and tile coordinates are stage positions in a caller-defined local
  reference frame;
- axis direction, machine units and controller coordinates remain outside the
  core;
- the core may propose future Z but never commands a stage.

## RG measurement result

The final result reports:

- measurement status and refusal reason;
- R/G displacement and uncertainty;
- proposed signed Z correction and bounded/clamped state;
- selected tissue patches and QC metrics;
- calibration identity and processing implementation identity;
- diagnostic counters safe for publication.

A refusal is a normal result. The OpenFlexure adapter decides whether to use
WHITE autofocus, acquire another RG anchor or stop according to GPL workflow
policy.

## Calibration result

The service returns an immutable calibration bundle containing:

- schema version and calibration identity;
- input-manifest digest;
- optics/mechanics identifiers supplied by the operator;
- flat-field and spectral profile references;
- shift-to-Z model and valid domain;
- residual, uncertainty and quality statistics;
- creation implementation and dependency versions.

The bundle contains no camera credentials, patient identifiers or arbitrary
host paths.

## Focus surface

Anchors are generic `(x_um, y_um, z_um)` observations with source, confidence
and timestamp metadata. Fitting returns serializable surface state and QC.
Prediction returns either a bounded Z estimate with support metrics or a refusal
reason. The core does not decide when a stage should move.

## Route planning

Inputs are field dimensions/overlap, calibrated travel bounds, a region or
mask artifact, requested traversal and focus-anchor policy. Output is a
deterministic ordered list of neutral actions:

- `visit` — capture candidate at XYZ/XY;
- `focus_anchor` — request a focus observation;
- `row_turn` — explicit traversal boundary;
- `component_start` — disconnected-region boundary;
- `skip` — optional recorded exclusion with reason.

The adapter converts neutral positions into machine commands and owns all
motion safety checks.

## Stitching

The input tile manifest contains immutable tile artifacts, physical stage
coordinates, pixel size, channel metadata and output requirements. The current
service returns only after producing a pyramidal OME-BigTIFF plus registration
and resource-usage reports. The GPL adapter starts the one-shot request in its
own process session; cancelling the OpenFlexure action terminates the owned
process group so neither core nor worker is orphaned.

The v1 production profile uses full-correlation registration. Expected-overlap
strip correlation is advertised only as an experimental capability and cannot
be selected unless the client explicitly opts into experimental operations.

## Cancellation and cleanup

- synchronous requests carry `timeout_ms`; the client enforces its wait bound;
- asynchronous persistence/status is reserved and is not advertised in this
  checkpoint;
- cancellation does not delete validated completed outputs;
- incomplete outputs are never reported as completed artifacts;
- the OpenFlexure adapter remains responsible for light-off and motion cleanup;
- the core performs no hardware cleanup because it controls no hardware.

## Security and privacy

- local-only binding by default;
- no shell command strings in the protocol;
- no arbitrary plugin/module loading;
- strict schemas with unknown-field rejection;
- allowlisted roots and normalized file URIs;
- bounded image dimensions, tile counts, memory budgets and worker counts;
- structured logs omit image content and specimen identifiers;
- published fixtures must be synthetic or privacy-reviewed.

## Licensing boundary tests

The release must include automated checks that:

- `fast-ofm-core` does not import `openflexure_microscope_server` or
  `labthings_fastapi`;
- `openflexure-wsi` does not import or vendor `fast_ofm_core`;
- the adapter passes protocol fixtures against a fake service;
- the core passes the same fixtures without OpenFlexure installed;
- distributions and containers do not silently combine the two source trees.

## Open decisions

- retention duration for job results and diagnostics;
- commercial deployment authentication outside localhost.
