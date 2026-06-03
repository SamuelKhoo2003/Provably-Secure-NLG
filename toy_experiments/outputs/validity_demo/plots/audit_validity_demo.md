# validity_demo synthetic validity stress-test audit

CSV used: toy_experiments/outputs/validity_demo/results/benchmark_results.csv
Generator: validity_demo
Scope: artificial controlled validity-only demo, not a natural language benchmark.
K values: [96]
N values: [32]
L values: [3, 9, 27]
T values: [50]

## Main plotted budgets
Certified-fraction plotted budget points: 30

Plain DPA max-token phrase blocker by L:
- L=3: 3
- L=9: 5
- L=27: 5

TPA max-token phrase blocker values by L:
- L=3: 4
- L=9: 7
- L=27: 8

Shared shard-aware MILP full sequence values by L:
- L=3: 5
- L=9: 21
- L=27: 57

## Token diagnostics
Per-token vote count examples by L, row 0:
- L=3: j=0:0,2,0,0,0,0,9,1,11,0,0,0,2,1,0,0,1,0,0,0,0,1,0,0,1,0,0,0,0,0,11,0,0,11,12,0,0,0,0,0,1,1,0,0,1,10,10,0,10,0;j=1:2,0,0,2,2,1,7,0,0,2,1,1,0,2,0,8,2,8,1,1,2,2,3,1,0,4,2,2,1,1,2,1,0,2,1,8,2,1,1,1,5,2,2,0,2,1,1,1,3,2;j=2:1,0,0,2,0,1,0,1,0,1,0,0,0,1,0,0,1,0,1,11,0,0,0,0,1,0,1,11,0,1,1,0,12,0,0,0,10,0,11,1,0,10,1,10,0,0,1,1,1,3
- L=9: j=0:1,0,1,0,4,3,0,3,0,9,3,1,1,0,2,0,4,0,1,0,2,0,1,0,1,2,4,1,2,4,3,0,8,3,1,2,1,4,2,1,8,0,0,2,1,2,3,0,4,1;j=1:2,1,1,1,1,3,2,1,2,2,2,0,2,1,5,8,0,8,1,0,2,3,3,3,1,0,2,0,1,4,0,1,2,5,1,2,2,2,2,1,0,4,2,3,1,1,0,3,1,1;j=2:1,0,0,2,2,0,0,11,1,10,2,0,0,1,12,1,0,0,0,0,0,0,11,1,10,0,1,11,0,2,0,0,0,0,0,0,0,0,0,3,11,1,0,0,0,1,0,0,0,1
- L=27: j=0:1,2,0,4,2,1,0,0,0,5,1,12,1,2,2,2,2,1,2,2,2,2,0,3,0,4,3,1,5,1,4,4,1,2,1,1,1,1,1,0,2,2,1,1,3,1,1,1,2,3;j=1:2,2,3,0,4,0,7,1,0,1,2,6,1,6,1,2,1,2,3,2,4,3,1,2,1,3,0,1,1,9,0,0,0,1,7,1,0,0,0,1,1,1,1,0,2,0,1,2,5,2;j=2:0,0,1,2,8,0,0,0,2,9,0,0,1,0,0,0,0,1,0,0,2,1,0,7,0,7,0,1,0,0,0,10,0,0,0,7,0,0,0,0,0,0,10,8,0,0,9,10,0,0

Plain DPA token radii by L, row 0:
- L=3: 5,4,5
- L=9: 5,4,5,5,4,3,3,4,4
- L=27: 5,3,4,5,4,4,5,5,5,4,4,4,4,5,5,5,5,4,5,5,4,3,5,3,5,3,5

TPA token radii by L, row 0:
- L=3: 8,7,8
- L=9: 7,5,8,9,5,4,4,6,6
- L=27: 5,4,7,7,5,5,5,8,8,6,6,5,5,7,8,8,9,5,7,5,6,4,8,5,5,5,5

Plain DPA token radii summary by L:
- L=3: min=2,p25=3,median=4,mean=4.04167,p75=5,max=5
- L=9: min=2,p25=4,median=4,mean=4.20486,p75=5,max=5
- L=27: min=2,p25=4,median=4,mean=4.17824,p75=5,max=5

TPA token radii summary by L:
- L=3: min=2,p25=5,median=6,mean=6.11458,p75=8,max=9
- L=9: min=2,p25=5,median=7,mean=6.22917,p75=8,max=9
- L=27: min=2,p25=5,median=6,mean=6.20486,p75=8,max=9

Plain DPA phrase radii summary by L:
- L=3: min=3,p25=4,median=5,mean=4.5625,p75=5,max=5
- L=9: min=5,p25=5,median=5,mean=5,p75=5,max=5
- L=27: min=5,p25=5,median=5,mean=5,p75=5,max=5

TPA phrase radii summary by L:
- L=3: min=4,p25=6,median=8,mean=7.125,p75=8,max=9
- L=9: min=7,p25=8,median=8,mean=8.1875,p75=8.25,max=9
- L=27: min=8,p25=8,median=9,mean=8.65625,p75=9,max=9

Shared MILP per-row phrase radii summary by L:
- L=3: min=5,median=11.5,mean=11.25,max=17
- L=9: min=21,median=27,mean=27.2812,max=33
- L=27: min=57,median=71.5,mean=71.5938,max=89

Cell percentages where TPA compares to Plain DPA by L:
- L=3: TPA>Plain 91.7%, TPA=Plain 8.33%, TPA<Plain 0%
- L=9: TPA>Plain 87.8%, TPA=Plain 12.2%, TPA<Plain 0%
- L=27: TPA>Plain 88.8%, TPA=Plain 11.2%, TPA<Plain 0%

Phrase-level TPA max-token radii by L, row 0:
- L=3: 8
- L=9: 9
- L=27: 9

Shard group assignments by token position, row 0:
- L=3: 0,1,2,3,4,5,6,7,8,9;3,4,5,6,7,8,9,10,11,12;6,7,8,9,10,11,12,13,14,15
- L=9: 0,1,2,3,4,6,7,8,9,69;3,5,6,7,8,9,10,11,12,65;6,7,8,9,10,11,12,13,14,15;9,10,11,12,14,15,16,17,18,92;0,1,2,12,13,14,15,16,17,18;15,16,17,18,19,20,21,22,23,24;18,19,20,21,22,23,24,25,26,27;21,22,23,24,25,26,27,29,30,92;24,25,26,27,28,29,30,31,32,33
- L=27: 0,1,2,3,4,5,6,7,8,9;3,4,5,6,7,8,9,10,11,12;0,1,6,7,8,9,10,11,12,13;9,10,11,12,13,14,15,16,17,18;0,12,14,15,16,17,18,19,20,89;15,16,17,18,19,20,21,22,23,89;0,1,15,18,19,20,21,22,23,89;21,22,23,24,25,26,27,28,29,30;24,25,26,27,28,29,30,31,32,89;0,26,27,28,29,30,31,32,33,92;0,30,31,32,33,34,35,36,37,38;34,35,36,37,38,39,40,41,42,91;36,37,38,39,41,42,43,44,45,92;0,38,39,40,41,42,43,44,45,46;0,42,43,44,45,46,47,48,49,93;45,46,47,48,49,50,51,52,53,54;48,49,50,51,52,53,55,56,57,88;51,52,53,54,55,56,57,58,59,60;0,54,55,56,57,58,59,60,61,91;57,58,59,60,61,62,63,64,65,66;0,1,58,60,61,62,63,64,65,66;63,64,65,66,67,68,69,70,71,72;0,63,64,66,67,68,69,70,71,72;69,70,71,72,73,74,75,76,77,78;72,73,74,75,76,77,78,79,80,81;76,77,78,79,80,81,82,83,84,91;78,79,80,82,83,84,85,86,87,88

Union size of required shard groups by L:
- L=3: 16
- L=9: 37
- L=27: 93

## MILP diagnostics
Shared MILP q1 status by L:
- L=3: OPTIMAL
- L=9: OPTIMAL
- L=27: OPTIMAL

Shared MILP q1 value by L:
- L=3: 5
- L=9: 21
- L=27: 57

Joint minus TPA gap by L:
- L=3: 1
- L=9: 14
- L=27: 49

Shared MILP relative lift over TPA by L:
- L=3: +1 mean budget units (25%)
- L=9: +14 mean budget units (200%)
- L=27: +49 mean budget units (612.5%)

Shared MILP mean-radius minus TPA mean-radius by L:
- L=3: 4.125
- L=9: 19.0938
- L=27: 62.9375

Shared MILP mean-radius relative lift over TPA by L:
- L=3: +4.125 mean budget units (57.8947%)
- L=9: +19.0938 mean budget units (233.206%)
- L=27: +62.9375 mean budget units (727.076%)

TPA minus plain DPA count-margin gap by L:
- L=3: 1
- L=9: 2
- L=27: 3

TPA relative lift over plain DPA by L:
- L=3: +1 mean budget units (33.3333%)
- L=9: +2 mean budget units (40%)
- L=27: +3 mean budget units (60%)

## Fail-fast confirmations
Budget curves monotone non-increasing: True
No MILP values silently dropped: True
Shared MILP q1 status OPTIMAL for every plotted L: True
Expected Plain DPA max-token phrase blocker < TPA max-token phrase blocker < Shared shard-aware MILP ordering observed: True
Expected gap observed: True
Gap grows with L: True

Generated cells individually feasible under intended shard group by L:
- L=3: True
- L=9: True
- L=27: True

## Explanation
validity_demo is artificial and controlled. It is not intended to model a natural language distribution.
Plain DPA max-token phrase blocker only reads the top-vs-target count margin at each token, so it misses the cost of overtaking many tied competitors.
TPA is count-based and sees each harmful target token as individually cheap after targeted count transfer.
The full shared MILP is shard-aware and must use one shared poisoned-shard allocation across target positions.
The demo assigns cheap target-token attacks to different shard groups, so the full harmful sequence requires more shared poisoned shards than TPA's count-only sequence baseline suggests.