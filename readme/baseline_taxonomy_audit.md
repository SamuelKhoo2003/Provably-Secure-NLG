# Baseline Taxonomy Audit

This audit uses the report-facing `toy_certificate/` implementation as the
active code path. It keeps existing CSV column names for compatibility and maps
them onto the baseline taxonomy below.

## Source-of-Truth Taxonomy

- Stability external baseline: DPA weakest-token stability.
- Validity token-level diagnostic/reference: DPA target-token validity.
- Validity main full-sequence baseline: TPA sequence validity.
- Proposed-method variants: Row-only shared MILP, column-only shared MILP, and
  joint row-column shared MILP.
- Diagnostics only: independent composition, atomic phrase aggregation, legacy
  raw DPA columns, and `runner_up` stability mode.

Interpretation rules:

- Stability and validity are separate objectives.
- Stability asks whether the clean output can be changed.
- Validity asks whether a specified harmful target output can be forced.
- Row-level stability is controlled by the weakest clean token because changing
  any one token destabilises the row.
- DPA weakest harmful-token validity is token-level. Aggregating it by the
  easiest harmful token is diagnostic/reference information, not a full-sequence
  validity baseline.
- TPA sequence validity is the main count-based full-sequence validity baseline.
  It is controlled by the hardest harmful target token because every harmful
  target token must be forced.
- Row-only, column-only, and joint row-column MILPs are proposed-method variants
  or ablations. The joint row-column shared MILP is the main proposed method.
- TPA is not used as a stability baseline in the active report-facing plots.
- Report-facing stability should use `stability_competitor_mode=all`.

## CSV Column Mapping

Stability external baseline family:

- `dpa_stab_cell_min`
- `dpa_stab_row_radius_q1`
- `dpa_stab_row_radius_qN`

Validity token-level diagnostic/reference:

- `dpa_val_cell_min`
- `dpa_val_row_weak_q1`
- `dpa_val_row_weak_qN`

Validity main full-sequence baseline:

- `tpa_val_cell_min`
- `tpa_val_sequence_q1`
- `tpa_val_sequence_qN`
- `tpa_val_sequence_mean`

Proposed-method variants:

- `row_stability`
- `column_stability`
- `row_col_stab_q1_r1`
- `row_col_stab_q1_rL`
- `row_col_stab_qN_r1`
- `row_col_stab_qN_rL`
- `row_validity`
- `column_validity_full_column`
- `row_col_val_q1`
- `row_col_val_qN`

Diagnostics only:

- `independent_*`
- `phrase_dpa_*`
- `phrase_independent_*`
- `raw_dpa_*`
- `runner_up` stability mode and related diagnostic CSV columns

## Audit Findings

- `toy_certificate/baselines.py` computes columns consistent with the taxonomy:
  DPA stability radii, DPA target-token validity diagnostics, TPA sequence
  validity baselines, and diagnostic independent/phrase references are separate.
- `toy_certificate/experiments.py` main comparison plots already keep TPA on
  validity plots only and label row/column/joint MILPs as proposed methods or
  ablations.
- The generated `audit_baseline_vs_milp_mapping.txt` classifies DPA weakest
  harmful-token validity as diagnostic in the main comparison path.
- Generic benchmark/budget labels previously collapsed DPA stability and DPA
  validity under `DPA weakest-token baseline`. They have been split so stability
  uses `DPA weakest-token stability baseline` and validity DPA uses
  `DPA weakest harmful-token diagnostic`.
- README wording now describes DPA weakest harmful-token validity as diagnostic
  only, not a full-sequence validity baseline.
- No active `toy_certificate/` plot label was found that uses TPA as a stability
  baseline.
- No Evaluation chapter file was found in the active repository tree checked
  for this audit. `README.md` is the only README-style document found outside
  ignored reference material.

## Main Plot Diagnostic Policy

- Independent composition and atomic phrase aggregation should stay out of main
  baseline plots. They may appear only in diagnostic plots or audit text.
- DPA weakest harmful-token validity may appear as a clearly labelled diagnostic
  overlay, but it should not be described as a full-sequence validity baseline.
- Row-only and column-only shared MILPs are ablations/proposed-method variants,
  not external baselines.

## Recommended Fixes

- Keep existing CSV column names stable and use this mapping for report text,
  plot legends, and generated audits.
- Use `DPA weakest-token stability baseline` for `dpa_stab_*` row/cell
  stability summaries.
- Use `DPA weakest harmful-token diagnostic` or `DPA target-token validity
  diagnostic` for `dpa_val_*` summaries.
- Use `TPA sequence baseline` only for validity sequence summaries.
- Continue filtering report-facing stability results to
  `stability_competitor_mode=all`; keep `runner_up` as a diagnostic override.
