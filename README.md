# Code-mixing and score compression in LLM routing

Does the RouteLLM BERT router score Hinglish questions differently from their English twins?
Main finding (152 confirmatory pairs): the average score is the same, but Hinglish scores have about half the spread of English scores, so routing depends on the cut-off. Answer quality was not measured.

## Layout
| Folder | Contents |
| --- | --- |
| `data/processed/` | `confirmatory_v1.csv`, the frozen 152 pairs (IDs 101-252). Never edit. |
| `data/provenance/` | Earlier drafts of the frozen file. Not for scoring. |
| `scores/` | `scores_confirmatory_v1.csv`, `scores_triple_v1.csv`, `scores_ood_scrambled_v1.csv` (confirmatory run). [Other files: pilot / exploratory, see below.] |
| `features/` | `features_confirmatory_v1.csv`, `features_triple_v1.csv`. [Others: pilot.] |
| `configs/` | `config_conf_v1.json` (router, version, file hash), `config_v4.json` (the five cut-offs). |
| `results/person3/` | Confirmatory analysis outputs, `_v2` set (the one the paper cites). |
| `docs/` | `correction_model_plan_v1.md` (signed plan) and the paper replacement text. |
| `results/correction/` | Final correction-model outputs (frozen, hashes in `HASHES.md`). |
| `results/correction_exploratory/` | Post-hoc checks by group and without Coding. Exploratory only. |
| `notebooks/` | Dataset building, scoring and analysis notebooks (outputs stripped). |
| `routerbench/` | Why-BERT experiment on 997 RouterBench prompts. Scores only, no prompt text. |
| `HASHES.md` | SHA-256 of every frozen file. Check with `python scripts/verify_hashes.py`. |

## Rules
- Frozen files are never edited. A fix becomes a new version with a new hash and a new run.
- Scoring ran once on the frozen file. The analysis notebook ran more than once (`_v1`, `_v2`); the paper cites `_v2`.
- Cut-off shares (10/20/30/50/70%) are nominal, from Chatbot Arena calibration.

## Status
- Pilot (80 pairs), confirmatory (152 pairs) and third set (200 pairs) are done and scored with BERT and MF.
- Final correction model is done. The plan (`docs/correction_model_plan_v1.md`) was signed on 15 Nov 2026 before any test score was read. The model was chosen on the third set and tested once on the 152 pairs. Result: the spread target was narrowly missed (SD ratio 1.93 to 1.13), the routing and ranking targets were met, but a training-free share-matched cut-off agrees with English routing better than the learned model (0.895 vs 0.863). Details are in `results/correction/` and `results/correction_exploratory/`.
- Still in progress: analysis of the third set and MF scores, paper text fixes, release.
- The earlier pilot-trained correction demo is withdrawn and must not be reported (tag `pre-cleanup` keeps it).

## Not included
ARC and LeetCode question text is not to be released; release will use pair IDs, source IDs and our own Hinglish text.

## results/person3/thirdset/
- All analyses in this folder are exploratory. No plan document exists for the third set.
- routing_mf_v1.csv uses fallback quantile thresholds, not the MF router’s calibration. Do not cite it.
- The MF crossover in summary_mf_v1.csv is not meaningful, because the slope is about [finish: value and why, e.g. close to 1 or near 0, which makes the crossover point undefined or unstable].