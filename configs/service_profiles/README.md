# Service profiles

Service profiles are measured properties of one serving class:

- model and tokenizer;
- GPU type;
- vLLM version;
- dtype;
- maximum model length, sequences, and batched tokens;
- KV block size;
- APC, Chunked Prefill, and eager-mode settings.

Generate profiles locally in each region with
`scripts/calibrate_local_service_profile.py`. Place deployment-specific
files under `configs/service_profiles/local/`; that directory is ignored
because profiles include endpoint paths and machine fingerprints.

The Router reads rates only from schema-v2 profile files referenced by the
topology. Formal runs reject inline Prefill/Decode constants.
