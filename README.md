code mixing llm router research analysis 

## Person 4: correction and safety evaluation

These analyses use saved scores; confirmatory prompts were not rescored.

- Integrated confirmatory check: 152 pairs; MAE 0.0719; 50% route agreement 0.8947. The detector classified all English and Hinglish prompts in the pairs as expected.
- English-only detector check: 280 prompts across GSM8K (100), MMLU (100), and MT-Bench (80); zero false positives and zero route changes.
- Hard-versus-Easy AUC: the multivariate linear model reached 0.645 (95% bootstrap CI 0.555–0.730). The other model CIs included 0.50.
- Replication verdict: score compression replicated in the confirmatory set; the pilot’s mean-score direction did not.

Saved in Drive under `ResearchPaper(ML)/correction_model_results/`:
`english_safety_summary_v1.csv`, `english_safety_prompt_audit_v1.csv`, `hard_easy_auc_bootstrap_v1.csv`, and `difficulty_analysis.csv`.

The detector identified 30 of 32 Hinglish prompts from three writers (93.75%). It missed two prompts. This is a small outside-writer check, not a broad estimate of detector accuracy.

### Outside-writer Hinglish validation

An additional detector validation was conducted using
32 naturally written Hinglish prompts from 3 external
writers.

- Total prompts: 32
- Correctly detected as Hinglish: 30
- False negatives: 2
- Detection rate: 93.75%
- False negative rate: 6.25%

The results indicate promising detection performance
on this small external sample. However, the sample
size is limited and does not establish broad
generalization.

Results are saved under:
`correction_model_results/`

Files:
- outside_hinglish_prompt_audit_v1.csv
- outside_hinglish_summary_v1.csv
- outside_hinglish_by_writer_v1.csv
