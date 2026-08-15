# data - frozen evaluation workloads

Formal experiments read these versioned, fixed-size files directly:

- `lmsys_first500.json`: the first 500 records of JITServe's flat LMSYS trace.
- `burst_first500.json`: the same 500 LMSYS requests with the first 500
  BurstGPT arrival timestamps.
- `deepresearch_flat_first500.json`: DeepResearch branches flattened by
  arrival time and truncated to the first 500 branch requests.

`first500_manifest.json` records their hashes and selection semantics.
Regenerate all four files with `scripts/build_first500_workloads.py`.

The DeepResearch selection intentionally matches the legacy request-level
experiments: it includes early branches from many workflows and may omit later
stages. Request-level SLO and latency metrics remain available, but Task SLO is
not defined for this trace and must be reported as `-`.

LMSYS is already a flat request trace, so taking its first 500 records does not
split workflows. Burst uses the same LMSYS request content and only replaces
arrival timestamps, so it has the same property.

## Source traces

The source traces come from the JITServe artifact:
https://github.com/UIUC-MLSys/JITServe

```bash
bash data/fetch_traces.sh
python scripts/build_first500_workloads.py
```

## Redistribution note

The LMSYS-derived files preserve upstream natural-language request text byte
for byte so their published SHA-256 digests and measured tokenization remain
reproducible. Generic credential scanners may flag security-related words or
password-like examples inside that workload text. No RAVEL endpoint credential,
API key, SSH password, private key, or environment secret is stored in this
repository. Review the upstream JITServe/LMSYS terms before redistributing the
workload files.
