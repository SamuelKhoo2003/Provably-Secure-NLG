# Sweep benchmark audit

- Source CSV: `toy_experiments/outputs/sweep_benchmark/L/results/benchmark_results.csv`
- Sweep: `L`
- Rows: 9
- Expected varied parameter(s): L
- Fixed vocabulary size `T`: yes

## Parameter values

- `K`: [20]
- `N`: [4]
- `L`: [4, 6, 8, 10, 12, 14, 16, 18, 20]
- `T`: [5]
- `delta_stab`: [0.2]
- `delta_val`: [0.2]
- `target_bias`: [0.3]
- `seed`: [0]

## Generated artifacts

- `sweep_L_stability_certificate_vs_L.pdf`
- `sweep_L_validity_certificate_vs_L.pdf`
- `sweep_L_runtime_vs_L.pdf`

## Methods and metrics

- Included: ['DPA weakest-token stability (`dpa_stab_row_radius_qN`)', 'Shared MILP, one token per row (`row_col_stab_qN_r1`)', 'Shared MILP, full token grid (`row_col_stab_qN_rL`)', 'Plain DPA validity (`plain_dpa_val_sequence_qN`)', 'TPA multi-sample validity (`tpa_val_sequence_qN`)', 'Shared MILP validity (`row_col_val_qN`)', 'Total Gurobi objective runtime (`runtime_gurobi_total`)']
- Missing or non-numeric: none

## Solver statuses

- `row_col_stab_q1_r1_status`: ['OPTIMAL']
- `row_col_stab_q1_rL_status`: ['OPTIMAL']
- `row_col_stab_qN_r1_status`: ['OPTIMAL']
- `row_col_stab_qN_rL_status`: ['OPTIMAL']
- `row_col_val_q1_status`: ['OPTIMAL']
- `row_col_val_qN_status`: ['OPTIMAL']

## Checks

- Unexpectedly varied parameters: none
- Expected varied parameters with fewer than two values: none
- Skipped series or plots: none
