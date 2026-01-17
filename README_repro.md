# Mondrian Map Reproducibility Guide

## Minimal case-study reproduction

```bash
python -m cli reproduce-case-study \
  --tp data/glass_TP.csv \
  --r1 data/glass_R1.csv \
  --r2 data/glass_R2.csv \
  --out outputs/case_study_glass \
  --force
```

Expected outputs:

- `outputs/case_study_glass/profiles/{aggressive,non_aggressive,baseline}/{R1_vs_TP,R2_vs_TP}/attributes.csv`
- `outputs/case_study_glass/profiles/{aggressive,non_aggressive,baseline}/{R1_vs_TP,R2_vs_TP}/relations.csv`
- `outputs/case_study_glass/figures/mondrian_panel_3x2.html`
- `outputs/case_study_glass/run_metadata.json`
- `outputs/case_study_glass/cache/pager/requests_manifest.jsonl`

## Streamlit layout (manual check)

1. Run the app (if desired):
   ```bash
   streamlit run app.py
   ```
2. Confirm the detailed view renders a square Mondrian map and dataset statistics appear in a separate row below the plot.
