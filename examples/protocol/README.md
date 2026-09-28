# Protocol v1 reference fixtures

These synthetic JSON fixtures are deterministic examples of the process
boundary. They contain no microscope address, specimen image, patient data or
machine calibration.

Run either request from an installed package:

```sh
fast-ofm-core run-request examples/protocol/v1/planning-route.request.json
fast-ofm-core run-request examples/protocol/v1/focus-surface.request.json
```

The corresponding `*.response.json` files are exact canonical responses from
the release candidate. `tests/test_protocol_examples.py` validates each request
against the dispatcher and checks byte-independent JSON equality with the
recorded response.

The JSON protocol fixtures are licensed under Apache-2.0 so independently
licensed clients can reuse them. This explanatory document is licensed under
CC-BY-NC-SA-4.0.

