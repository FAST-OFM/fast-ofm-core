# Licensing

## Original Fast OFM implementation

Files under `src/fast_ofm_core/` and `tests/`, except protocol schemas, are
licensed under `PolyForm-Noncommercial-1.0.0` once their provenance rows are
approved. Commercial use requires a separate written license.

## Protocol schemas

Files under `src/fast_ofm_core/protocol/v1/` are licensed under `Apache-2.0`.
This narrow grant covers the neutral interoperability schemas only. It does
not grant rights to the PolyForm-covered algorithm implementation.

## Excluded implementation

This repository must not contain:

- OpenFlexure Microscope Server GPL source or derived planner/server code;
- modified or copied `openflexure-stitching` LGPL implementation;
- private specimen images or identifying metadata;
- code of unknown ownership.

An optional LGPL stitching worker and the GPL OpenFlexure adapter are separate
distributions communicating over the process protocol.
