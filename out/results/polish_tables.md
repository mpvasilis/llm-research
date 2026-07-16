Source: `out/results/summary_v3.json`

### Emission rates: counts and Wilson 95% CIs (primary threshold 0.82)

| model | behavior | advice k/n (rate, CI) | factual k/n (rate, CI) |
|---|---|---|---|
| 1B | empathy_opener | 35/60 (0.583, [0.457,0.699]) | 1/30 (0.033, [0.006,0.167]) |
| 1B | validation | 22/60 (0.367, [0.256,0.493]) | 0/30 (0.0, [0,0.114]) |
| 1B | disclaimer | 54/60 (0.9, [0.799,0.953]) | 2/30 (0.067, [0.018,0.213]) |
| 1B | crisis_referral | 52/60 (0.867, [0.758,0.931]) | 0/30 (0.0, [0,0.114]) |
| 1B | structure | 60/60 (1.0, [0.94,1.0]) | 17/30 (0.567, [0.392,0.726]) |
| 7B | empathy_opener | 37/60 (0.617, [0.49,0.729]) | 1/30 (0.033, [0.006,0.167]) |
| 7B | validation | 23/60 (0.383, [0.271,0.51]) | 0/30 (0.0, [0,0.114]) |
| 7B | disclaimer | 55/60 (0.917, [0.819,0.964]) | 2/30 (0.067, [0.018,0.213]) |
| 7B | crisis_referral | 52/60 (0.867, [0.758,0.931]) | 0/30 (0.0, [0,0.114]) |
| 7B | structure | 58/60 (0.967, [0.886,0.991]) | 16/30 (0.533, [0.361,0.698]) |

### Threshold sensitivity: advice emission at e5 thresholds 0.80 / 0.82 / 0.84

| model | behavior | 0.80 | 0.82 | 0.84 |
|---|---|---|---|---|
| 1B | empathy_opener | 0.94 | 0.59 | 0.297 |
| 1B | validation | 0.85 | 0.363 | 0.107 |
| 1B | disclaimer | 0.997 | 0.9 | 0.473 |
| 1B | crisis_referral | 0.997 | 0.867 | 0.4 |
| 1B | structure | 1.0 | 1.0 | 0.997 |
| 7B | empathy_opener | 0.94 | 0.617 | 0.37 |
| 7B | validation | 0.817 | 0.383 | 0.18 |
| 7B | disclaimer | 0.997 | 0.913 | 0.52 |
| 7B | crisis_referral | 0.997 | 0.86 | 0.38 |
| 7B | structure | 1.0 | 0.97 | 0.95 |

### Stagewise emergence: advice emission at primary threshold 0.82

| model | stage | empathy | validation | disclaimer | crisis_referral | structure |
|---|---|---|---|---|---|---|
| 1B | base | 0.217 | 0.073 | 0.483 | 0.42 | 0.757 |
| 1B | sft | 0.393 | 0.167 | 0.697 | 0.717 | 0.847 |
| 1B | dpo | 0.55 | 0.38 | 0.933 | 0.87 | 1.0 |
| 1B | rlvr | 0.567 | 0.41 | 0.923 | 0.867 | 1.0 |
| 1B | instruct | 0.59 | 0.363 | 0.9 | 0.867 | 1.0 |
| 7B | base | 0.23 | 0.087 | 0.577 | 0.477 | 0.853 |
| 7B | sft | 0.493 | 0.183 | 0.767 | 0.74 | 0.787 |
| 7B | dpo | 0.597 | 0.33 | 0.923 | 0.877 | 0.977 |
| 7B | instruct | 0.617 | 0.383 | 0.913 | 0.86 | 0.97 |

### Final model emission by condition at primary threshold 0.82

| model | behavior | advice | factual | emotional | neutral_advice | domain_factual |
|---|---|---|---|---|---|---|
| 1B | empathy_opener | 0.59 | 0.04 | 0.79 | 0.02 | 0.34 |
| 1B | validation | 0.363 | 0.007 | 0.12 | 0.0 | 0.04 |
| 1B | disclaimer | 0.9 | 0.08 | 0.51 | 0.74 | 0.82 |
| 1B | crisis_referral | 0.867 | 0.0 | 0.7 | 0.33 | 0.38 |
| 1B | structure | 1.0 | 0.58 | 0.82 | 1.0 | 0.92 |
| 7B | empathy_opener | 0.617 | 0.027 | 0.76 | 0.07 | 0.36 |
| 7B | validation | 0.383 | 0.013 | 0.13 | 0.03 | 0.06 |
| 7B | disclaimer | 0.913 | 0.06 | 0.54 | 0.72 | 0.82 |
| 7B | crisis_referral | 0.86 | 0.007 | 0.61 | 0.34 | 0.28 |
| 7B | structure | 0.97 | 0.547 | 0.83 | 1.0 | 0.84 |

### Prompt-clustered permutation tests: advice vs control conditions

| model | behavior | control | diff | perm_p |
|---|---|---|---|---|
| 1B | empathy_opener | factual | 0.55 | 0.00019998000199980003 |
| 1B | empathy_opener | emotional | -0.2 | 0.0096990300969903 |
| 1B | empathy_opener | neutral_advice | 0.57 | 0.00019998000199980003 |
| 1B | empathy_opener | domain_factual | 0.25 | 0.015898410158984102 |
| 1B | validation | factual | 0.357 | 0.00019998000199980003 |
| 1B | validation | emotional | 0.243 | 0.0026997300269973002 |
| 1B | validation | neutral_advice | 0.363 | 0.00019998000199980003 |
| 1B | validation | domain_factual | 0.323 | 0.0034996500349965005 |
| 1B | disclaimer | factual | 0.82 | 0.00019998000199980003 |
| 1B | disclaimer | emotional | 0.39 | 0.00019998000199980003 |
| 1B | disclaimer | neutral_advice | 0.16 | 0.006199380061993801 |
| 1B | disclaimer | domain_factual | 0.08 | 0.3524647535246475 |
| 1B | crisis_referral | factual | 0.867 | 0.00019998000199980003 |
| 1B | crisis_referral | emotional | 0.167 | 0.0112988701129887 |
| 1B | crisis_referral | neutral_advice | 0.537 | 0.00019998000199980003 |
| 1B | crisis_referral | domain_factual | 0.487 | 0.00019998000199980003 |
| 1B | structure | factual | 0.42 | 0.00019998000199980003 |
| 1B | structure | emotional | 0.18 | 0.00019998000199980003 |
| 1B | structure | neutral_advice | 0.0 | 1.0 |
| 1B | structure | domain_factual | 0.08 | 0.0183981601839816 |
| 7B | empathy_opener | factual | 0.59 | 0.00019998000199980003 |
| 7B | empathy_opener | emotional | -0.143 | 0.0978902109789021 |
| 7B | empathy_opener | neutral_advice | 0.547 | 0.00019998000199980003 |
| 7B | empathy_opener | domain_factual | 0.257 | 0.0306969303069693 |
| 7B | validation | factual | 0.37 | 0.00019998000199980003 |
| 7B | validation | emotional | 0.253 | 0.0027997200279972004 |
| 7B | validation | neutral_advice | 0.353 | 0.00029997000299970003 |
| 7B | validation | domain_factual | 0.323 | 0.004399560043995601 |
| 7B | disclaimer | factual | 0.853 | 0.00019998000199980003 |
| 7B | disclaimer | emotional | 0.373 | 0.00019998000199980003 |
| 7B | disclaimer | neutral_advice | 0.193 | 0.0011998800119988001 |
| 7B | disclaimer | domain_factual | 0.093 | 0.15878412158784122 |
| 7B | crisis_referral | factual | 0.853 | 0.00019998000199980003 |
| 7B | crisis_referral | emotional | 0.25 | 0.0004999500049995 |
| 7B | crisis_referral | neutral_advice | 0.52 | 0.00019998000199980003 |
| 7B | crisis_referral | domain_factual | 0.58 | 0.00019998000199980003 |
| 7B | structure | factual | 0.423 | 0.00019998000199980003 |
| 7B | structure | emotional | 0.14 | 0.004199580041995801 |
| 7B | structure | neutral_advice | -0.03 | 0.573942605739426 |
| 7B | structure | domain_factual | 0.13 | 0.025997400259974 |

### Per-phrase SFT-assistant recoverability (role-scoped)

Per-million uses each model's exact SFT assistant-document count when available. `infl` = whole-conversation (blob) count / assistant-only count.

| model | words | phrase | assistant | per-million | infl |
|---|---|---|---|---|---|
| 1B | 1 | `first,` | 44422 | 51287.4 | 1.19 |
| 1B | 2 | `healthcare professional` | 3075 | 3550.2 | 1.05 |
| 1B | 2 | `professional help` | 1765 | 2037.8 | 1.03 |
| 1B | 2 | `seek professional` | 1353 | 1562.1 | 1.02 |
| 1B | 2 | `consult a` | 891 | 1028.7 | 1.1 |
| 1B | 2 | `it's understandable` | 218 | 251.7 | 1.04 |
| 1B | 3 | `here are some` | 11914 | 13755.3 | 1.11 |
| 1B | 3 | `mental health professional` | 2109 | 2434.9 | 1.04 |
| 1B | 3 | `it's natural to` | 152 | 175.5 | 1.03 |
| 1B | 4 | `here are a few` | 5421 | 6258.8 | 1.12 |
| 1B | 4 | `it sounds like you` | 1130 | 1304.6 | 1.03 |
| 1B | 4 | `steps you can take` | 792 | 914.4 | 1.01 |
| 1B | 4 | `it's okay to feel` | 159 | 183.6 | 1.02 |
| 1B | 4 | `your feelings are valid` | 49 | 56.6 | 1.0 |
| 1B | 4 | `it's normal to feel` | 33 | 38.1 | 1.03 |
| 7B | 2 | `healthcare professional` | 3106 | 3306.6 | 1.07 |
| 7B | 2 | `professional help` | 1774 | 1888.6 | 1.04 |
| 7B | 2 | `seek professional` | 1355 | 1442.5 | 1.02 |
| 7B | 2 | `consult a` | 927 | 986.9 | 1.11 |
| 7B | 3 | `here are some` | 12049 | 12827.0 | 1.11 |
| 7B | 3 | `i'm really sorry` | 2369 | 2522.0 | 1.0 |
| 7B | 3 | `mental health professional` | 2120 | 2256.9 | 1.04 |
| 7B | 3 | `talk to a` | 286 | 304.5 | 1.43 |
| 7B | 3 | `it's natural to` | 153 | 162.9 | 1.03 |
| 7B | 3 | `it's completely normal` | 78 | 83.0 | 1.32 |
| 7B | 4 | `here are a few` | 5511 | 5866.9 | 1.12 |
| 7B | 4 | `it sounds like you` | 1139 | 1212.5 | 1.04 |
| 7B | 4 | `steps you can take` | 803 | 854.9 | 1.01 |
| 7B | 4 | `i'm sorry to hear` | 322 | 342.8 | 1.05 |
| 7B | 4 | `your feelings are valid` | 50 | 53.2 | 1.0 |
| 7B | 4 | `it's normal to feel` | 33 | 35.1 | 1.03 |

**Distinct-phrase leave-one-out (3-word behavioral phrases, per-model-denominator per-million):**
- 1B: over 3 distinct 3-word phrases, dropping the top generic phrase `here are some` lowers the mean from 5455.2 to 1305.2 per-million.
- 7B: over 6 distinct 3-word phrases, dropping the top generic phrase `here are some` lowers the mean from 3026.0 to 1065.9 per-million.
