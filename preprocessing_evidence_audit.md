# Preprocessing Evidence Audit

## Verified Dataset Totals

- Successfully processed floor-level samples: 4,566
- Retained samples: 3,761
- Excluded samples: 805
- Retained proportion: 82.37%
- Excluded proportion: 17.63%
- Minimum retained encoded connected-region count: 3
- Encoded count range before filtering: 0-32
- Mean encoded count before filtering: 5.3574
- Median encoded count before filtering: 5

The reconciliation checks are complete: no retained sample is missing from the clean folder, no excluded sample is present in the clean folder, and no extra clean samples were found.

Verified source files are stored under:

```text
evidence/preprocessing/
```

## Exclusion Distribution

| Encoded connected-region count | Excluded samples | Share of excluded set | Share of all processed samples |
|---:|---:|---:|---:|
| 0 | 243 | 30.19% | 5.32% |
| 1 | 187 | 23.23% | 4.10% |
| 2 | 375 | 46.58% | 8.21% |
| **Total** | **805** | **100%** | **17.63%** |

## Interpretation of Representative Examples

The retained evidence shows that the excluded set is mixed. Some samples are empty, wall-only, incomplete or very low-complexity. Other samples contain visually meaningful room and structural regions but remain below the threshold because the connected-component encoding can merge adjacent semantic regions.

The threshold should therefore be described as a practical preprocessing heuristic rather than an architectural rule.

The evidence supports the following interpretation:

> The threshold reduced very low-complexity and potentially uninformative samples, but it may also have removed some valid small or strongly connected layouts. Its effect on the final data distribution is therefore a recognised limitation.

## Required Dissertation Terminology

Use **encoded connected-region count** rather than **room count** when describing the preprocessing label. The implementation key is named `room_count`, but the value is produced through eight-connected component analysis of the combined non-background and non-wall semantic mask.

Similarly, the implementation key `outline` stores the filled binary floor-plan support mask used by the model; it should not be described as only an external boundary line.

## Methodology Wording Supported by the Evidence

> The preprocessing pipeline produced 4,566 semantic floor-plan samples. A practical threshold of at least three encoded connected regions was applied, retaining 3,761 samples and excluding 805 samples. The excluded set comprised 243 samples with a count of zero, 187 with a count of one and 375 with a count of two. Representative inspection showed that the filter removed empty, incomplete and very low-complexity layouts, although some plausible small layouts were also excluded because the connected-component encoding did not correspond directly to architectural room instances. The threshold is therefore treated as a heuristic preprocessing decision rather than an architectural standard.

## Optional Appendix Evidence

If representative preprocessing exclusions are included in a dissertation appendix, use a small selection with:

- sample ID;
- encoded connected-region count;
- exclusion reason;
- semantic mask; and
- binary support mask.

The main chapter does not need to reproduce all 805 excluded records. If no separate appendix is included, the report should not refer to a non-existent appendix; the retained repository evidence is sufficient for reproducibility and audit purposes.
