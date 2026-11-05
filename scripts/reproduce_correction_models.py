from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, cohen_kappa_score
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests

BASE = Path(__file__).resolve().parent
OUTPUT_DIR = BASE / "correction_model_results_reproduced"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ID_COL = "pair_id"
EN_SCORE = "english_score"
HI_SCORE = "hinglish_score"
FEATURE_COLS = [
    "hinglish_tokens",
    "hinglish_words",
    "hinglish_fertility"
]

# Load the existing score and feature files. This script does not rescore prompts.
pilot_scores = pd.read_csv(BASE / "scores" / "scores_v2.csv")
pilot_features = pd.read_csv(BASE / "features" / "features_v2.csv")
test_scores = pd.read_csv(BASE / "scores" / "scores_confirmatory_v1.csv")
test_features = pd.read_csv(BASE / "features" / "features_confirmatory_v1.csv")

pilot_features = pilot_features.rename(columns={
    "hi_tokens": "hinglish_tokens",
    "hi_words": "hinglish_words",
    "hi_fertility": "hinglish_fertility"
})

pilot = pilot_scores.merge(
    pilot_features[[ID_COL] + FEATURE_COLS],
    on=ID_COL,
    how="inner",
    validate="one_to_one"
)
test = test_scores.merge(
    test_features[[ID_COL] + FEATURE_COLS],
    on=ID_COL,
    how="inner",
    validate="one_to_one"
)

assert len(pilot) == 80, f"Expected 80 pilot pairs, found {len(pilot)}"
assert len(test) == 152, f"Expected 152 confirmatory pairs, found {len(test)}"

# Fit correction models on the pilot set.
X_train = pilot[[HI_SCORE] + FEATURE_COLS]
y_train = pilot[EN_SCORE]
X_test = test[[HI_SCORE] + FEATURE_COLS]
y_test = test[EN_SCORE]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

multi_model = LinearRegression().fit(X_train_scaled, y_train)
gb_model = GradientBoostingRegressor(
    n_estimators=30,
    learning_rate=0.03,
    max_depth=1,
    random_state=42
).fit(X_train_scaled, y_train)

uni_model = LinearRegression().fit(
    pilot[[HI_SCORE]].to_numpy(),
    y_train.to_numpy()
)

# Generate confirmatory predictions.
original = test[HI_SCORE].to_numpy()
constant = np.full(len(test), y_train.mean())
multi = np.clip(multi_model.predict(X_test_scaled), 0, 1)
orthogonal = np.clip(
    (original - pilot[HI_SCORE].mean())
    * (pilot[EN_SCORE].std() / pilot[HI_SCORE].std())
    + pilot[EN_SCORE].mean(),
    0,
    1
)
gradient_boosting = np.clip(gb_model.predict(X_test_scaled), 0, 1)
univariate = np.clip(
    uni_model.predict(test[[HI_SCORE]].to_numpy()),
    0,
    1
)

predictions = {
    "Original RouteLLM": original,
    "Constant Baseline": constant,
    "Linear (score + token features)": multi,
    "Orthogonal Rescaling": orthogonal,
    "Gradient Boosting": gradient_boosting,
    "Linear (univariate)": univariate
}

thresholds = {
    "10%": 0.5949,
    "20%": 0.5094,
    "30%": 0.4651,
    "50%": 0.4066,
    "70%": 0.3372
}

# Model comparison metrics.
result_rows = []
for model_name, scores in predictions.items():
    row = {
        "Model": model_name,
        "MAE": mean_absolute_error(y_test, scores),
        "English SD": np.std(y_test, ddof=1),
        "Corrected SD": np.std(scores, ddof=1)
    }
    row["SD Ratio"] = row["English SD"] / row["Corrected SD"]

    for threshold_name, threshold in thresholds.items():
        english_route = (y_test.to_numpy() >= threshold).astype(int)
        predicted_route = (scores >= threshold).astype(int)
        row[f"{threshold_name} Agreement"] = np.mean(
            english_route == predicted_route
        )
        row[f"{threshold_name} Kappa"] = cohen_kappa_score(
            english_route, predicted_route
        )

    result_rows.append(row)

results_df = pd.DataFrame(result_rows)

# Seeded paired bootstrap: positive improvement means lower MAE than original.
rng = np.random.default_rng(0)
bootstrap_rows = []
english = y_test.to_numpy()

for model_name, scores in predictions.items():
    if model_name == "Original RouteLLM":
        continue

    differences = (
        np.abs(english - original)
        - np.abs(english - np.asarray(scores))
    )
    indices = rng.integers(
        0, len(differences), size=(10000, len(differences))
    )
    bootstrap_means = differences[indices].mean(axis=1)

    bootstrap_rows.append({
        "Model": model_name,
        "MAE improvement": differences.mean(),
        "95% CI lower": np.percentile(bootstrap_means, 2.5),
        "95% CI upper": np.percentile(bootstrap_means, 97.5)
    })

bootstrap_df = pd.DataFrame(bootstrap_rows)

# English-vs-Hinglish route discordances and Holm correction across thresholds.
route_rows = []
for model_name, scores in predictions.items():
    for threshold_name, threshold in thresholds.items():
        english_route = y_test.to_numpy() >= threshold
        hinglish_route = np.asarray(scores) >= threshold

        english_only_strong = int(np.sum(english_route & ~hinglish_route))
        hinglish_only_strong = int(np.sum(~english_route & hinglish_route))
        both_strong = int(np.sum(english_route & hinglish_route))
        both_weak = int(np.sum(~english_route & ~hinglish_route))

        table = [
            [both_strong, english_only_strong],
            [hinglish_only_strong, both_weak]
        ]

        route_rows.append({
            "Model": model_name,
            "Threshold": threshold_name,
            "English-only strong": english_only_strong,
            "Hinglish-only strong": hinglish_only_strong,
            "McNemar p": mcnemar(table, exact=True).pvalue
        })

route_results_df = pd.DataFrame(route_rows)
route_results_df["Holm-adjusted p"] = (
    route_results_df.groupby("Model")["McNemar p"]
    .transform(lambda p: multipletests(p, method="holm")[1])
)

# Save reproducibility outputs separately; existing files remain untouched.
results_df.to_csv(OUTPUT_DIR / "model_comparison_v2.csv", index=False)

corrected_df = test[[ID_COL, EN_SCORE, HI_SCORE]].copy()
for model_name, scores in predictions.items():
    corrected_df[model_name] = scores
corrected_df.to_csv(OUTPUT_DIR / "corrected_scores_v2.csv", index=False)

route_results_df.to_csv(OUTPUT_DIR / "mcnemar_holm_v2.csv", index=False)
bootstrap_df.to_csv(OUTPUT_DIR / "bootstrap_improvements_v2.csv", index=False)

print("Pilot pairs:", len(pilot))
print("Confirmatory pairs:", len(test))
print("Univariate intercept:", uni_model.intercept_)
print("Univariate slope:", uni_model.coef_[0])
print("Reproduced files saved in:", OUTPUT_DIR)
