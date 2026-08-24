# Campaign-B mechanism decision and RAVEL upgrade

Date: 2026-08-24

## Decision

Production RAVEL-Unified now uses WorkYield placement and CenterYield dispatch
for both completion-only and mixed traffic. The protected OnTimeSet cohort and
Router-side soft-admission withholding were removed.

The following mechanisms remain because Campaign-B did not ablate them:
causal completion-pressure detection, the held-out semantic output profile,
risk-aware quotes, pre-KV revisable placement, completion virtual-service
accounting, and engine priority support.

## Evidence

| Variant | Runs | SLO | TTFT mean | E2E mean | Queue wait |
|---|---:|---:|---:|---:|---:|
| Legacy Full | reused 5-run mean | 63.04% | 8.519 s | 16.094 s | 7.893 s |
| NoSoftAdmission | 1 | 73.40% | 2.271 s | 12.042 s | 1.609 s |
| NoOnTimeSet | 1 | 75.00% | 1.946 s | 11.655 s | 1.011 s |
| Upgraded Full | 1 | 79.00% | 2.276 s | 10.786 s | 1.193 s |

Every row used the same 500 request IDs, 216,313 prompt tokens, 190,498 output
tokens, trace seed 42, physical-network topology, five endpoints, and cache
reset protocol. The upgraded run recorded zero soft-admission active requests,
zero protected requests, and zero nonnegative soft plan generations.

Decode TPOT in the ablations and upgraded run was slower than the legacy Full,
so the SLO gain is not explained by a faster decode engine. It comes with a
large reduction in queue wait.

## Reproducibility

- Frozen evidence source: /root/autodl-tmp/RAVEL-Campaigns-16x-20260823
- Upgraded source: /root/autodl-tmp/RAVEL-upgraded-20260824
- Campaign-B evidence: /root/autodl-tmp/Campaign-B-ablation-20260824
- Legacy source SHA256: 1017bb90e5ceda2d3b915abccd3bd704876770ab3f4b733a741a4c9b06ad6e4e
- Upgraded GPU-run source SHA256: b6205a4318f585b8865aa494444644d5df5af01e69ab19895f9c776a4d4bac8d

The GPU-run source hash binds the code used for the upgraded run. Later
documentation cleanup may change a whole-tree digest but does not change
scheduler behavior.
