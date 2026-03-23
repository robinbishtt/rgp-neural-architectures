# Pre-computed Results Tables

These CSV files contain the key quantitative results from the paper,
pre-computed from full experiments. Reviewers who cannot run the full
pipeline can verify the numbers directly from these files.

| File | Contents | Paper Reference |
|---|---|---|
| table1_h3_architecture_comparison.csv | Architecture comparison, all metrics | Table 1 |
| table2_h1_r2_by_width_seed.csv | H1 R² values (4 widths × 10 seeds) | Sec 5.1 |
| table3_h2_lmin_by_xi.csv | H2 L_min measurements | Sec 5.2 |
| table4_h2_statistical_summary.csv | H2 complete statistical analysis | Sec 5.2 |
| table5_h3_statistical_tests.csv | H3 Welch t-test + Cohen's d | Sec 5.3 |
| table6_ablation_activations.csv | Activation function ablation | Appendix G.1 |
| table7_ablation_initialization.csv | Initialization ablation | Appendix G.1 |
| table8_width_ablation_finite_size.csv | Finite-width corrections | Appendix F.1 |

## Verification

To reproduce these tables from scratch:
```bash
bash reproduce.sh          # full run (~24-72 hours)
bash reproduce.sh --fast-track  # approximate values (~5 minutes)
```
