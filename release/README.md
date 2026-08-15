# Frozen RAVEL release provenance

This public bundle contains only RAVEL artifacts.

- `recorded-inputs/ravel-final/`: exact final-v3 topology and deployment check.
- `CALIBRATION_MANIFEST.json`: the strict 32/256/128/5 calibration contract,
  RAVEL profile paths and SHA-256 digests.
- `RELEASE_MANIFEST.json`: source, result, topology and workload freeze metadata.
- `SHA256SUMS`: content digests for all tracked release files except itself.

The final-v3 per-cell manifests retain the paths recorded on the testbed. Use
the portable topology under `configs/topologies/` in a new checkout. Every
measured cell records source digest
`c9108ee504f900153c7ec0a071c51d771e76a399fcb5e55ccf93a88c6794661b`.

No baseline result, profile, topology, report, or comparison figure is included.
