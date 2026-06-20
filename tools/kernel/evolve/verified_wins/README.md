# Verified wins from `tools/kernel/evolve/`

This directory archives every kernel diff that has been **discovered by
the evolve loop AND survived all rigorous verification gates**:

1. **Numerics** — dtype-aware allclose + cosine floor + anti-cheat
2. **Critic** — Opus 4.7 adversarial refutation
3. **Paired-t stats** — N≥8 rounds, p<0.05
4. **Cross-shape** — at least one production shape clears p<0.05 with no
   statistically significant regressions on the others

Diffs are saved here with their evidence JSON/MD so any future engineer
can audit the result and the methodology. **A diff in this directory has
NOT necessarily been merged** — it has been verified worth merging. The
human reviewer makes the final call.

## Index

| Date | Kernel | Diff | Cross-shape result | Notes |
|---|---|---|---|---|
| 2026-06-20 | `ragged_paged_attention/v3` | [rpa_v3_exp_m_diff_cast.diff](rpa_v3_exp_m_diff_cast.diff) | 2 wins, 0 regressions, 2 ties; mean +2.27% across 4 production shapes ([json](rpa_v3_exp_m_diff_cast_cross_shape.json), [md](rpa_v3_exp_m_diff_cast_cross_shape.md)) | Discovered by shape-aware multi-shape evolution (`rpa_v3_multishape_evolve.py`). Llama-3-8B +5.08% p=0.005, Qwen3-0.6B-long +3.17% p=0.013. fp8 KV shape is the only soft point (-1.4%, NS). |

## How to add an entry

After the ship pipeline (`tools/kernel/evolve/ship_pipeline.py`) reports
overall=PASS:

```bash
cp /tmp/<workdir>/winner.diff tools/kernel/evolve/verified_wins/<kernel>_<short_descr>.diff
cp /tmp/<workdir>/cross_shape.json tools/kernel/evolve/verified_wins/<kernel>_<short_descr>_cross_shape.json
cp /tmp/<workdir>/cross_shape.md tools/kernel/evolve/verified_wins/<kernel>_<short_descr>_cross_shape.md
# Append a row to the Index table above.
git add tools/kernel/evolve/verified_wins/ && git commit ...
```
