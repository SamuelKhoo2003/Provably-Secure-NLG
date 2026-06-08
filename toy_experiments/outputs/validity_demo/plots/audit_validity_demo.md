# validity_demo synthetic validity stress-test audit

CSV used: toy_experiments/outputs/validity_demo/results/benchmark_results.csv
Generator: validity_demo
Scope: artificial controlled validity-only demo, not a natural language benchmark.
K values: [20, 24]
N values: [1, 2]
L values: [2, 3, 4, 5]
T values: [8]

## Main plotted budgets
Certified-fraction plotted budget points: 21

Plain DPA max-token phrase blocker by L:
- L=2: 2.5
- L=3: 2.5
- L=4: 3
- L=5: 3

TPA max-token phrase blocker values by L:
- L=2: 3.5
- L=3: 3.5
- L=4: 4
- L=5: 4

Shared shard-aware MILP full sequence values by L:
- L=2: 4.25
- L=3: 4.5
- L=4: 6.5
- L=5: 6.5

## Token diagnostics
Per-token vote count examples by L, row 0:
- L=2: j=0:4,1,4,4,1,5,1,0;j=1:5,0,1,2,5,1,5,1
- L=3: j=0:4,1,4,4,1,5,1,0;j=1:5,0,1,2,5,1,5,1;j=2:2,5,0,2,2,1,6,2
- L=4: j=0:4,1,4,4,1,5,1,0;j=1:5,0,1,2,5,1,5,1;j=2:2,5,0,2,2,1,6,2
- L=5: j=0:4,1,4,4,1,5,1,0;j=1:5,0,1,2,5,1,5,1;j=2:2,5,0,2,2,1,6,2

Plain DPA token radii by L, row 0:
- L=2: 2,2
- L=3: 2,2,2
- L=4: 2,2,2,3
- L=5: 2,2,2,3,2

TPA token radii by L, row 0:
- L=2: 3,3
- L=3: 3,3,3
- L=4: 3,3,3,4
- L=5: 3,3,3,4,3

Plain DPA token radii summary by L:
- L=2: min=2,p25=2,median=2,mean=2,p75=2,max=2
- L=3: min=2,p25=2,median=2,mean=2,p75=2,max=2
- L=4: min=2,p25=2,median=2,mean=2.25,p75=2.25,max=3
- L=5: min=2,p25=2,median=2,mean=2.2,p75=2,max=3

TPA token radii summary by L:
- L=2: min=3,p25=3,median=3,mean=3,p75=3,max=3
- L=3: min=3,p25=3,median=3,mean=3,p75=3,max=3
- L=4: min=3,p25=3,median=3,mean=3.25,p75=3.25,max=4
- L=5: min=3,p25=3,median=3,mean=3.2,p75=3,max=4

Plain DPA phrase radii summary by L:
- L=2: min=2,p25=2,median=2,mean=2,p75=2,max=2
- L=3: min=2,p25=2,median=2,mean=2,p75=2,max=2
- L=4: min=3,p25=3,median=3,mean=3,p75=3,max=3
- L=5: min=3,p25=3,median=3,mean=3,p75=3,max=3

TPA phrase radii summary by L:
- L=2: min=3,p25=3,median=3,mean=3,p75=3,max=3
- L=3: min=3,p25=3,median=3,mean=3,p75=3,max=3
- L=4: min=4,p25=4,median=4,mean=4,p75=4,max=4
- L=5: min=4,p25=4,median=4,mean=4,p75=4,max=4

Shared MILP per-row phrase radii summary by L:
- L=2: min=4,median=4,mean=4,max=4
- L=2: min=4,median=5,mean=5,max=6
- L=2: min=5,median=5,mean=5,max=5
- L=2: min=4,median=4.5,mean=4.5,max=5
- L=3: min=4,median=4,mean=4,max=4
- L=3: min=4,median=5.5,mean=5.5,max=7
- L=3: min=5,median=5,mean=5,max=5
- L=3: min=5,median=6,mean=6,max=7
- L=4: min=7,median=7,mean=7,max=7
- L=4: min=7,median=8,mean=8,max=9
- L=4: min=6,median=6,mean=6,max=6
- L=4: min=6,median=6.5,mean=6.5,max=7
- L=5: min=7,median=7,mean=7,max=7
- L=5: min=7,median=8,mean=8,max=9
- L=5: min=6,median=6,mean=6,max=6
- L=5: min=6,median=7.5,mean=7.5,max=9

Cell percentages where TPA compares to Plain DPA by L:
- L=2: TPA>Plain 100%, TPA=Plain 0%, TPA<Plain 0%
- L=3: TPA>Plain 100%, TPA=Plain 0%, TPA<Plain 0%
- L=4: TPA>Plain 100%, TPA=Plain 0%, TPA<Plain 0%
- L=5: TPA>Plain 100%, TPA=Plain 0%, TPA<Plain 0%

Phrase-level TPA max-token radii by L, row 0:
- L=2: 3
- L=3: 3
- L=4: 4
- L=5: 4

Shard group assignments by token position, row 0:
- L=2: 0,1,2,3,4,5,6,7;0,1,3,4,5,6,7,8
- L=3: 0,1,2,3,4,5,6,7;0,1,3,4,5,6,7,8;0,1,6,8,9,10,11,12
- L=4: 0,1,2,3,4,5,6,7;0,1,3,4,5,6,7,8;0,1,6,8,9,10,11,12;0,9,10,11,12,13,15,16
- L=5: 0,1,2,3,4,5,6,7;0,1,3,4,5,6,7,8;0,1,6,8,9,10,11,12;0,9,10,11,12,13,15,16;4,12,13,14,15,16,17,18

Union size of required shard groups by L:
- L=2: 9
- L=3: 13
- L=4: 16
- L=5: 19

## MILP diagnostics
Shared MILP q1 status by L:
- L=2: OPTIMAL
- L=3: OPTIMAL
- L=4: OPTIMAL
- L=5: OPTIMAL

Shared MILP q1 value by L:
- L=2: 4.25
- L=3: 4.5
- L=4: 6.5
- L=5: 6.5

Joint minus TPA gap by L:
- L=2: 0.75
- L=3: 1
- L=4: 2.5
- L=5: 2.5

Shared MILP relative lift over TPA by L:
- L=2: +0.75 mean budget units (21.4286%)
- L=3: +1 mean budget units (28.5714%)
- L=4: +2.5 mean budget units (62.5%)
- L=5: +2.5 mean budget units (62.5%)

Shared MILP mean-radius minus TPA mean-radius by L:
- L=2: 0.5
- L=3: 2
- L=4: 2.5
- L=5: 3

Shared MILP mean-radius relative lift over TPA by L:
- L=2: +0.5 mean budget units (12.5%)
- L=3: +2 mean budget units (50%)
- L=4: +2.5 mean budget units (62.5%)
- L=5: +3 mean budget units (66.6667%)

TPA minus plain DPA count-margin gap by L:
- L=2: 1
- L=3: 1
- L=4: 1
- L=5: 1

TPA relative lift over plain DPA by L:
- L=2: +1 mean budget units (40%)
- L=3: +1 mean budget units (40%)
- L=4: +1 mean budget units (33.3333%)
- L=5: +1 mean budget units (33.3333%)

## Fail-fast confirmations
Budget curves monotone non-increasing: True
No MILP values silently dropped: True
Shared MILP q1 status OPTIMAL for every plotted L: True
Expected Plain DPA max-token phrase blocker < TPA max-token phrase blocker < Shared shard-aware MILP ordering observed: False
Expected gap observed: True
Gap grows with L: True

Generated cells individually feasible under intended shard group by L:
- L=2: True
- L=3: True
- L=4: True
- L=5: True

## Explanation
validity_demo is artificial and controlled. It is not intended to model a natural language distribution.
Plain DPA max-token phrase blocker only reads the top-vs-target count margin at each token, so it misses the cost of overtaking many tied competitors.
TPA is count-based and sees each harmful target token as individually cheap after targeted count transfer.
The full shared MILP is shard-aware and must use one shared poisoned-shard allocation across target positions.
The demo assigns cheap target-token attacks to different shard groups, so the full harmful sequence requires more shared poisoned shards than TPA's count-only sequence baseline suggests.