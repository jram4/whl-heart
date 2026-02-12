# whl-heart

WHL 2026 competition analysis project for Phase 1 (1a-1d).

## Contents

- `src/phase1_pipeline.py`: end-to-end pipeline for power rankings, matchup probabilities, line disparity, and visualization.
- `outputs/`: final order-robust outputs.
- `outputs_seq/`: sequential (single-order) comparison outputs.
- `whl_2025 - whl_2025(1).csv`: main dataset.
- `WHSDSC_Rnd1_matchups - matchups.csv`: round-1 matchup input file.
- `data science comp important.pdf`: assignment instructions.
- `WHSDSC 2026 Glossary.pdf`: glossary reference.

## Reproduce

```bash
source venv/bin/activate
python src/phase1_pipeline.py \
  --output-dir outputs \
  --order-robust \
  --tune-elo \
  --n-shuffles 400 \
  --tune-shuffles 60 \
  --matchups-csv "WHSDSC_Rnd1_matchups - matchups.csv"
```

## Key outputs

- `outputs/phase1a_power_rankings.csv`
- `outputs/phase1a_round1_matchup_probabilities.csv`
- `outputs/phase1b_top10_line_disparity.csv`
- `outputs/phase1c_disparity_vs_team_strength.png`
- `outputs/phase1_run_summary.json`
