# Preprocessing Evidence Audit

## Verified dataset totals

- Processed NPZ samples: 4,566
- Retained samples: 3,761
- Excluded samples: 805
- Retained proportion: 82.37%
- Excluded proportion: 17.63%
- Minimum retained encoded connected-region count: 3
- Encoded count range before filtering: 0-32
- Mean encoded count before filtering: 5.3574
- Median encoded count before filtering: 5

The reconciliation checks are complete: no retained sample is missing from the clean folder, no excluded sample is present in the clean folder, and no extra clean samples were found.

## Exclusion distribution

| Encoded connected-region count | Excluded samples | Share of excluded set | Share of all processed samples |
|---:|---:|---:|---:|
| 0 | 243 | 30.19% | 5.32% |
| 1 | 187 | 23.23% | 4.10% |
| 2 | 375 | 46.58% | 8.21% |
| **Total** | **805** | **100%** | **17.63%** |

## Interpretation of representative examples

The appendix-ready contact sheet confirms that the excluded set is mixed. Some samples are empty, wall-only, highly incomplete or very low-complexity. Other samples contain visually meaningful room and structural regions but remain below the threshold because the connected-component encoding merges adjacent semantic regions. Therefore, the threshold should be described as a practical heuristic rather than an architectural rule.

The evidence supports the following conclusion:

> The threshold reduced very low-complexity and potentially uninformative samples, but it may also have removed some valid small or strongly connected layouts. Its effect on the final data distribution is therefore a recognised limitation.

## Required dissertation terminology

Use **encoded connected-region count** rather than **room count** when discussing the preprocessing label. The implementation key is named `room_count`, but the value is produced through 8-connected component analysis of the combined non-background and non-wall semantic mask.

## Recommended methodology wording

> The preprocessing pipeline produced 4,566 semantic floor-plan samples. A practical threshold of at least three encoded connected regions was applied, retaining 3,761 samples and excluding 805 samples. The excluded set comprised 243 samples with a count of zero, 187 with a count of one and 375 with a count of two. Representative inspection showed that the filter removed empty, incomplete and very low-complexity layouts, although some plausible small layouts were also excluded because the connected-component encoding did not correspond directly to architectural room instances. The threshold is therefore treated as a heuristic preprocessing decision rather than an architectural standard.

## Appendix recommendation

Include the contact sheet or 6-8 representative examples with:

- sample ID;
- encoded connected-region count;
- exclusion reason;
- semantic mask;
- binary support mask.

The main body should refer to the appendix but should not reproduce all 805 records.
