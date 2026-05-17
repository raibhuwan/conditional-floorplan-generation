# Experiment Log

This document records the major training experiments, evaluation results, and refinement observations obtained during development of the conditional semantic floor plan generation framework.

The experiments focus on:
- conditional U-Net baseline performance,
- adversarial cGAN training stability,
- dataset scaling behaviour,
- and lightweight post-generation refinement strategies.

## Metric Note

Validation IoU values reported during training correspond to epoch-based semantic segmentation validation performance.

Final reported mIoU values are obtained from the full post-training evaluation pipeline and are therefore not directly equivalent to training-stage validation IoU.

## Experiment Summary

| Experiment | Clean Dataset Size | Best Validation IoU |
|---|---:|---:|
| EXP-01 | 492 | 0.269 |
| EXP-02 | 492 | 0.278 |
| EXP-03 | 1004 | 0.319 |
| EXP-04 | 1966 | 0.373 |

The results show a consistent improvement in validation performance as dataset size and adversarial training stability increased across successive experiments.

## EXP-01 — Initial cGAN Training

### Configuration
- Requested source SVG samples: 500
- Dataset size: 492 samples
- Generator: U-Net
- Discriminator: PatchGAN
- Batch size: 2
- Epochs: 30
- Generator LR: 1e-4
- Discriminator LR: 5e-5
- Lambda CE: 20.0
- Lambda GAN: 0.1

### Results
- Best validation IoU: 0.269
- Training IoU (final): 0.624
- Validation IoU (final): 0.235

### Observations
- The discriminator rapidly became overconfident during training.
- Discriminator loss collapsed close to zero, indicating imbalance between generator and discriminator optimisation.
- GAN loss increased significantly, suggesting that the generator struggled to fool the discriminator.
- The model achieved reasonable semantic learning but showed signs of overfitting and poor generalisation.


---

## EXP-02 — Stabilised cGAN Training (492 Samples)

### Configuration
- Requested source SVG samples: 500
- Dataset size: 492 samples
- Generator: U-Net
- Discriminator: PatchGAN
- Batch size: 2
- Epochs: 30
- Generator LR: 1e-4
- Discriminator LR: 1e-5
- Lambda CE: 30.0
- Lambda GAN: 0.05
- Real label smoothing: 0.9

### Results
- Best validation IoU: 0.278
- Training IoU (final): 0.660
- Validation IoU (final): 0.243

### Observations
- Lower discriminator learning rate improved adversarial stability.
- Increasing CE weighting improved semantic consistency.
- Label smoothing prevented discriminator overconfidence.
- Validation performance improved slightly, although overfitting remained significant.


---

## EXP-03 — Increased Dataset Training (1004 Samples)

### Configuration
- Requested source SVG samples: 1000
- Dataset size: 1004 samples
- Generator: U-Net
- Discriminator: PatchGAN
- Batch size: 2
- Epochs: 30
- Generator LR: 1e-4
- Discriminator LR: 1e-5
- Lambda CE: 30.0
- Lambda GAN: 0.05
- Real label smoothing: 0.9

### Results
- Best validation IoU: 0.319
- Training IoU (final): 0.740
- Validation IoU (final): 0.313

### Observations
- Increasing dataset size improved validation performance and generalisation.
- Training became more stable with reduced discriminator dominance.
- The model learned stronger semantic structure while maintaining adversarial consistency.
- Overfitting remained present but was reduced compared to earlier experiments.
- Best checkpoint occurred around epoch 23.

---

## EXP-04 — Increased Dataset Training (1966 Samples)

### Configuration
- Requested source SVG samples: 2000
- Clean dataset size after filtering: 1966 samples
- Generator: U-Net
- Discriminator: PatchGAN
- Batch size: 2
- Epochs: 30
- Generator LR: 1e-4
- Discriminator LR: 1e-5
- Lambda CE: 30.0
- Lambda GAN: 0.05
- Real label smoothing: 0.9
- MAX_COUNT: 18

### Results
- Best validation IoU: 0.373
- Best checkpoint epoch: 21
- Final training IoU: 0.753
- Final validation IoU: 0.360

### Observations
- Increasing the dataset from 1004 to 1966 clean samples improved validation IoU from 0.319 to 0.373.
- Discriminator loss remained stable around 0.16–0.17, suggesting improved adversarial balance.
- Training IoU continued to increase, but the gap between training and validation performance shows that some overfitting remains.
- The results suggest that cGAN training benefits from larger layout diversity in the CubiCasa5K subset.

---

# Overall Findings

## Training Stability
Early adversarial training experiments showed severe discriminator dominance, where discriminator loss rapidly collapsed toward zero and generator adversarial loss increased substantially. This resulted in unstable optimisation and poor validation generalisation.

Reducing discriminator learning rate, increasing cross-entropy weighting, and applying label smoothing significantly improved training stability. These modifications produced more balanced adversarial learning behaviour and more consistent validation performance.

---

## Effect of Dataset Scaling

Increasing dataset size consistently improved validation performance:

| Clean Dataset Size | Best Validation IoU |
|---|---|
| 492 | 0.278 |
| 1004 | 0.319 |
| 1966 | 0.373 |

The results suggest that conditional adversarial floor plan generation benefits strongly from increased layout diversity and larger training distributions.

---

## Generator–Discriminator Balance

The experiments demonstrated that balancing supervised semantic learning and adversarial learning is critical for stable conditional floor plan generation.

High adversarial influence caused discriminator overconfidence and unstable optimisation, while stronger cross-entropy supervision improved semantic consistency and structural learning.

The final configuration achieved more stable discriminator behaviour, with discriminator loss remaining approximately within the range of 0.16–0.17 during later epochs.

---

## Generalisation Behaviour

Although validation performance improved substantially with larger datasets, a persistent gap remained between training and validation IoU scores, indicating continued overfitting.

However, the gap decreased as dataset size increased, suggesting that larger and more diverse training distributions improve generalisation capability.

---

## Architectural Observations

The cGAN framework demonstrated improved structural consistency and semantic layout quality compared to earlier unstable configurations. The combination of conditional U-Net generation with PatchGAN adversarial refinement provided a practical balance between segmentation accuracy and structural realism while remaining computationally feasible on Apple M1 hardware.

---

## Key Conclusion

The experiments demonstrate that lightweight conditional adversarial learning can improve semantic floor plan generation when combined with:
- stable encoder–decoder architectures,
- balanced adversarial optimisation,
- structured preprocessing,
- and sufficient dataset diversity.

These findings support the use of conditional GAN frameworks for improving spatial coherence and structural realism in automated floor plan generation.

## cGAN Evaluation on 1966 Clean Samples

### Checkpoint
- Checkpoint: `outputs/checkpoints/cgan_unet_patchgan_best.pt`
- Dataset size evaluated: 1966 samples

### Metrics
- Mean mIoU excluding background: 0.546
- Mean adjacency F1: 0.238
- Mean compactness: 0.547

### Observation
The cGAN achieved improved semantic overlap on the filtered dataset, with a mean mIoU of 0.546. The compactness score suggests that the generated regions maintain moderate geometric regularity, while the adjacency F1 remains lower, indicating that room relationship prediction is still challenging.

## Baseline vs cGAN Comparison (1966 Samples)

| Metric | U-Net Baseline | cGAN |
|---|---|---|
| Mean mIoU (no background) | 0.321 | 0.546 |
| Mean adjacency F1 | 0.179 | 0.238 |
| Mean compactness | 0.469 | 0.547 |

### Observations
- The cGAN achieved substantially higher semantic segmentation performance compared to the baseline U-Net model.
- Adjacency similarity improved under adversarial training, suggesting better preservation of relational room structure.
- Compactness scores also increased, indicating improved geometric regularity and reduced fragmentation in generated layouts.
- The results suggest that adversarial refinement contributes not only to semantic prediction accuracy but also to structural and spatial coherence.

## Qualitative Visual Comparison

Visual comparison between the baseline U-Net model and the cGAN model showed that adversarial training produced smoother and more spatially coherent semantic regions.

Compared to the baseline, the cGAN predictions exhibited:
- reduced fragmentation,
- larger connected room regions,
- smoother geometric structure,
- and improved semantic consistency.

However, both approaches still produced architecturally imperfect layouts due to the limited conditioning information provided to the models. Since generation was conditioned only on building outline masks and room-count parameters, the models were required to infer internal spatial organisation statistically without explicit relational or circulation constraints.

These observations support the quantitative findings, where the cGAN achieved higher compactness and adjacency similarity scores compared to the baseline U-Net model.

Example outputs are available in:

`outputs/comparison_samples/`

![Comparison Example](docs/images/comparison_example.png)

## cGAN + Morphological Refinement Evaluation

### Results
- Mean mIoU excluding background: 0.567
- Mean adjacency F1: 0.250
- Mean compactness: 0.660

### Observations
- Morphological refinement improved all three evaluation metrics compared to the raw cGAN output.
- Compactness increased substantially from 0.547 to 0.660, showing that the refinement step improved geometric regularity.
- Mean mIoU also increased from 0.546 to 0.567, indicating that lightweight refinement did not degrade semantic accuracy.
- Adjacency F1 improved slightly from 0.238 to 0.250, suggesting a modest improvement in relational layout consistency.
- These results support the project aim that lightweight post-generation refinement can improve spatial quality while preserving semantic consistency.

## Morphological Refinement Comparison

| Method | mIoU | Adjacency F1 | Compactness |
|---|---:|---:|---:|
| U-Net Baseline | 0.321 | 0.179 | 0.469 |
| U-Net + Morphology | 0.327 | 0.183 | 0.611 |
| cGAN | 0.546 | 0.238 | 0.547 |
| cGAN + Morphology | 0.567 | 0.250 | 0.660 |

### Observations
- Morphological refinement improved compactness for both U-Net and cGAN outputs.
- U-Net compactness improved from 0.469 to 0.611, showing that morphology helps reduce fragmented or irregular predicted regions.
- cGAN compactness improved from 0.547 to 0.660, giving the best geometric regularity among the tested methods.
- cGAN + morphology achieved the best overall performance across all three metrics.
- The results suggest that adversarial training improves semantic and relational structure, while morphology refinement mainly improves geometric compactness.

## cGAN + Hill-Climbing Refinement

### Results
- Mean mIoU excluding background: 0.443
- Mean adjacency F1: 0.228
- Mean compactness: 0.565

### Observations
- Hill-climbing refinement improved compactness slightly compared to the raw cGAN output.
- However, semantic segmentation accuracy decreased substantially, with mIoU dropping from 0.546 to 0.443.
- Adjacency similarity also decreased slightly compared to morphology refinement.
- The results suggest that compactness-only local optimisation may over-smooth semantic regions and degrade structural detail.
- Compared to hill-climbing refinement, morphology-based refinement achieved a better balance between geometric improvement and semantic preservation.

## Final Comparative Evaluation

| Method | mIoU | Adjacency F1 | Compactness |
|---|---:|---:|---:|
| U-Net Baseline | 0.321 | 0.179 | 0.469 |
| U-Net + Morphology | 0.327 | 0.183 | 0.611 |
| U-Net + Hill-Climb | 0.256 | 0.181 | 0.528 |
| cGAN | 0.546 | 0.238 | 0.547 |
| cGAN + Morphology | 0.567 | 0.250 | 0.660 |
| cGAN + Hill-Climb | 0.443 | 0.228 | 0.565 |

### Key Findings

- The conditional GAN significantly outperformed the baseline U-Net across all evaluation metrics.
- Morphological refinement consistently improved geometric compactness while largely preserving semantic accuracy.
- cGAN + morphology achieved the best overall performance, producing the highest mIoU, adjacency similarity, and compactness scores.
- Hill-climbing refinement improved compactness moderately but reduced semantic accuracy, particularly for the U-Net baseline.
- The results suggest that lightweight deterministic refinement methods provide a better balance between semantic preservation and geometric improvement than compactness-only optimisation strategies.

### Interpretation

The experiments indicate that adversarial learning improves semantic and relational layout structure, while lightweight morphology-based refinement improves geometric regularity and spatial coherence. However, optimisation-driven refinement based primarily on compactness may over-smooth room boundaries and reduce semantic fidelity.

These findings support the hypothesis that lightweight post-processing refinement can improve generated floor-plan quality without requiring significantly more complex generative architectures.

# Current Research Direction

Current work focuses on:
- improving spatial coherence,
- reducing overfitting,
- expanding dataset scale,
- and investigating lightweight refinement methods for semantic floor plan generation.