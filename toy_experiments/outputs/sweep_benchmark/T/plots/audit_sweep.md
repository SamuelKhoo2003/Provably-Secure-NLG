# Sweep benchmark audit

- Source CSV: `toy_experiments/outputs/sweep_benchmark/T/results/benchmark_results.csv`
- Sweep: `T`
- Rows: 35
- Expected varied parameter(s): T
- Vocabulary size `T`: varied as expected

## Parameter values

- `K`: [20]
- `N`: [4]
- `L`: [5]
- `T`: [3, 5, 7, 9, 11, 13, 15]
- `delta_stab`: [0.2]
- `delta_val`: [0.2]
- `target_bias`: [0.3]
- `seed`: [0, 10, 20, 30, 40]

## Generated artifacts

- `sweep_T_stability_certificate_vs_T.pdf`
- `sweep_T_validity_certificate_vs_T.pdf`
- `sweep_T_runtime_vs_T.pdf`

## Methods and metrics

- Included: ['DPA weakest-token stability (`dpa_stab_row_radius_qN`)', 'Shared MILP, one token per row (`row_col_stab_qN_r1`)', 'Shared MILP, full token grid (`row_col_stab_qN_rL`)', 'Plain DPA max token (`plain_dpa_val_sequence_qN`)', 'TPA max-token phrase baseline (`tpa_val_sequence_qN`)', 'Shared MILP validity (`row_col_val_qN`)', 'Total Gurobi objective runtime (`runtime_gurobi_total`)']
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
