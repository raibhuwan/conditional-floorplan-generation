# Condition-Sensitivity, Determinism and Runtime Evidence Audit

## Test Scope

The supplementary test used the selected epoch-20 U-Net checkpoint, the same filled binary floor-plan support mask, and five encoded connected-region count conditions: 3, 4, 5, 6 and 7. It evaluated raw predictions, morphology-refined predictions, repeated inference for determinism and local runtime on Apple MPS.

The implementation retains the historical command-line name `room_count`, but the tested value is the encoded connected-region count used by the final dissertation.

Verified source files are stored under:

```text
evidence/condition_sensitivity/
```

## Verified Findings

### Sensitivity to the Encoded Count

All five requested count values produced different raw outputs and different morphology-refined outputs. Adjacent changes affected approximately 1.06% to 1.74% of raw pixels and 1.27% to 2.16% of morphology-refined pixels. This shows that the numerical condition influences the generated semantic mask.

However, the number of connected regions did not increase consistently with the requested condition:

| Requested encoded count | Raw predicted regions | Morphology predicted regions |
|---:|---:|---:|
| 3 | 5 | 2 |
| 4 | 4 | 2 |
| 5 | 3 | 2 |
| 6 | 4 | 2 |
| 7 | 4 | 2 |

The raw prediction matched the requested value only for the tested count of 4. The relationship was not monotonic: increasing the supplied count did not reliably increase the number of predicted connected regions. Morphology reduced every tested output to two connected regions, demonstrating that post-processing can substantially alter agreement with the count condition.

The test therefore supports the statement that the encoded count acts as a conditioning signal but not as a hard count constraint. It does not establish exact architectural room-count control.

### Determinism

Three repeated runs using the same support mask and encoded count of 4 produced exactly identical output arrays. Their SHA-256 hashes matched and zero pixels changed. The selected U-Net inference pipeline was therefore deterministic under the tested setup.

This also means that the current inference route does not provide multiple alternatives for the same checkpoint and identical input conditions.

### Runtime

After five warm-up runs, 30 timed runs produced:

- mean U-Net inference: 8.644 ms;
- median U-Net inference: 8.667 ms;
- 95th percentile inference: 8.904 ms;
- mean morphology processing: 3.269 ms;
- approximate combined compute time: 11.913 ms; and
- checkpoint loading: 195.410 ms.

The inference and morphology timings exclude image saving. The first individual run was slower because it included device warm-up effects and should not be used as the representative runtime.

## Dissertation Interpretation

The evidence supports a cautious statement that changing the encoded connected-region count changes the generated semantic mask, so the model does not completely ignore the numerical condition. However, the weak and non-monotonic relationship between requested and predicted connected-region counts shows that the condition does not provide reliable exact-count control.

The determinism evidence supports the limitation that repeated inference with the same checkpoint and conditions produces the same output. Runtime evidence can be used to describe the inference route as lightweight on the tested Apple M1 setup, provided it is made clear that the reported values represent local compute time rather than complete user-facing generation time.
