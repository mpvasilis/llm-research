# CPU-only de-risk preliminary

OLMo-2-1B-Instruct · 8 advice prompts vs 4 factual controls

## Q1 — Behavior emission (advice vs factual)

| behavior | advice | factual |
|---|---|---|
| empathy_opener | 0.125 | 0.0 |
| validation | 0.125 | 0.0 |
| disclaimer | 0.25 | 0.0 |
| crisis_referral | 0.0 | 0.0 |
| structure | 0.75 | 0.0 |

Behavioral coverage: advice **0.021** vs factual 0.0

## Q3 — SFT-assistant recoverability (behavioral vs topical control)

- behavioral phrases: advice **1.0** / factual 0
- topical control words: advice 1.0 / factual 1.0

## Q4 — Behavioral-phrase source domain

- forum-ish share: advice **0.531** vs factual 0.0
- novelty (1 - verbatim coverage): advice 0.632 / factual 0.728

## Q2 — Role-scoping inflation (blob vs assistant-only SFT counts)

| phrase | assistant | blob | inflation× |
|---|---|---|---|
| here are some | 3947 | 4451 | 1.13 |
| steps you can take | 217 | 220 | 1.01 |
| mental health professional | 142 | 153 | 1.08 |
| professional help | 139 | 151 | 1.09 |
| i'm really sorry | 58 | 63 | 1.09 |
| it's normal to feel | 8 | 8 | 1.0 |

*Inflation > 1 = blob count includes user turns; assistant-only is the model's true target signal.*