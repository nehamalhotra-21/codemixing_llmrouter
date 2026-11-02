code mixing llm router research analysis 

## Person 4: correction and safety evaluation

These analyses use saved scores; confirmatory prompts were not rescored.

- Integrated confirmatory check: 152 pairs; MAE 0.0719; 50% route agreement 0.8947. The detector classified all English and Hinglish prompts in the pairs as expected.
- English-only detector check: 280 prompts across GSM8K (100), MMLU (100), and MT-Bench (80); zero false positives and zero route changes.
- Hard-versus-Easy AUC: the multivariate linear model reached 0.645 (95% bootstrap CI 0.555–0.730). The other model CIs included 0.50.
- Replication verdict: score compression replicated in the confirmatory set; the pilot’s mean-score direction did not.

Saved in Drive under `ResearchPaper(ML)/correction_model_results/`:
`english_safety_summary_v1.csv`, `english_safety_prompt_audit_v1.csv`, `hard_easy_auc_bootstrap_v1.csv`, and `difficulty_analysis.csv`.
