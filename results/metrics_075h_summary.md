| method (0.75 h of training) | return ↑ | failed clips ↓ | body pos (cm) ↓ | body ori (°) ↓ | joints (°) ↓ | contact agreement ↑ | foot slip (cm/s) ↓ | penetration (mm) ↓ | root drift (cm) ↓ |
|---|---|---|---|---|---|---|---|---|---|
| SuperMimic (1 seed) | 0.775 | 4 | 5.29 | 12.5 | 6.15 | 0.727 | 8.44 | 0.066 | 116 |
| SuperMimic + EMA | 0.852 ± 0.002 | 0 | 4.60 ± 0.03 | 11.1 ± 0.2 | 5.61 ± 0.05 | 0.719 ± 0.001 | 8.00 ± 0.06 | 0.063 ± 0.003 | 54 ± 1 |
| InstantTrack, no contact reward | 0.873 ± 0.001 | 0 | 3.50 ± 0.08 | 7.7 ± 0.3 | 4.15 ± 0.02 | 0.738 ± 0.002 | 7.73 ± 0.08 | 0.057 ± 0.001 | 58 ± 3 |
| **InstantTrack** | 0.867 ± 0.001 | 0 | 3.53 ± 0.04 | 7.5 ± 0.1 | 4.56 ± 0.02 | 0.812 ± 0.000 | 6.62 ± 0.04 | 0.052 ± 0.001 | 61 ± 5 |
