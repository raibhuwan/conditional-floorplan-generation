# Condition-Sensitivity, Determinism and Runtime Evidence Audit

## Test scope

The test used the selected U-Net checkpoint from epoch 20, the same binary floor-plan support mask, and five encoded connected-region count conditions: 3, 4, 5, 6 and 7. It evaluated raw predictions, morphology-refined predictions, repeated inference for determinism, and local runtime on Apple MPS.

## Verified findings

### Sensitivity to the encoded count

All five requested count values produced different raw outputs and different morphology-refined outputs. Adjacent changes affected approximately 1.06% to 1.74% of raw pixels and 1.27% to 2.16% of morphology-refined pixels. This shows that the numerical condition influences the generated semantic mask.

However, the number of connected regions did not increase consistently with the requested condition:

| Requested encoded count | Raw predicted regions | Morphology predicted regions |
|---:|---:|---:|
| 3 | 5 | 2 |
| 4 | 4 | 2 |
| 5 | 3 | 2 |
| 6 | 4 | 2 |
| 7 | 4 | 2 |

The raw result matched the requested value only for the tested count of 4. The relationship was not monotonic: increasing the condition did not reliably increase the number of predicted connected regions. Morphology reduced every tested output to two connected regions, demonstrating that refinement can substantially alter condition alignment.

Therefore, this test supports the statement that the encoded count acts as a conditioning signal but not as a hard count constraint. It does not establish exact architectural room-count control.

### Determinism

Three repeated runs using the same support mask and encoded count of 4 produced exactly identical output arrays. Their SHA-256 hashes matched and zero pixels changed. The selected U-Net inference pipeline is therefore deterministic under the tested setup.

### Runtime

After five warm-up runs, 30 timed runs produced:

- mean U-Net inference: 8.644 ms
- median U-Net inference: 8.667 ms
- 95th percentile inference: 8.904 ms
- mean morphology processing: 3.269 ms
- approximate combined compute time: 11.913 ms
- checkpoint loading: 195.410 ms

The inference and morphology timings exclude image saving. The first individual run was slower because it included device warm-up effects and should not be used as the representative runtime.

## Report implications

The report can state that changing the encoded connected-region count produced distinct layouts, showing that the model did not completely ignore the numerical condition. However, the weak and non-monotonic relationship between requested and predicted connected-region counts demonstrates that the condition did not provide reliable count control. The deterministic output also means that the current architecture does not generate multiple alternatives for the same input and checkpoint. Runtime evidence supports describing the prototype as lightweight for local inference, while making clear that this is compute time rather than complete user-facing generation time.
