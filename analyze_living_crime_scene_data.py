# analyze_living_crime_scene_bestof.py
# ---------------------------------------------------------------------------
# "Best of both worlds" analyse voor Living Crime Scene SNN
#
# Auteur: Berry Kriesels
#
# Combineert:
#   - Klassieke stats: summary, z-scores t.o.v. baseline, Cohen's d
#   - Visualisaties: boxplots, scatter, Bland–Altman, heatmap, PCA, etc.
#   - ML: multi-class & binary classifier + feature importance
#   - Optionele anomaly detection (IsolationForest)
#
# Dit script pakt de CSV uit de simulatie en trekt die helemaal binnenstebuiten
# om te zien of we de aanvallen statistisch en visueel kunnen onderscheiden.
# ---------------------------------------------------------------------------

import os
import time
from datetime import timedelta
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_auc_score,
)
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# SciPy is handig voor de T-tests, maar als je het niet hebt geïnstalleerd
# crasht het script tenminste niet, we slaan dan gewoon de ANOVA over.
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Globale instellingen
# ---------------------------------------------------------------------------

RESULT_CSV = "living_crime_scene_results.csv"  # De input file (uit het andere script)
SUMMARY_CSV = "living_crime_scene_summary_stats.csv"
ZSCORES_CSV = "living_crime_scene_zscores_vs_baseline.csv"
EFFECT_SIZES_CSV = "living_crime_scene_effect_sizes_vs_baseline.csv"
CLASSIFIER_REPORT_TXT = "classifier_report.txt"
ENHANCED_REPORT_TXT = "enhanced_analysis_report.txt"

# Mappen aanmaken voor de plaatjes, wel zo netjes.
FIG_DIR = "figures"
ENH_FIG_DIR = "enhanced_figures"
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(ENH_FIG_DIR, exist_ok=True)

# Even de plot-stijl wat strakker zetten
plt.style.use("default")
sns.set_palette("husl")
plt.rcParams["figure.figsize"] = (8, 5)
plt.rcParams["axes.grid"] = True

# Nodig om delays te berekenen als ze niet in de CSV staan
BASELINE_DURATION_SEC = 5.0


# ---------------------------------------------------------------------------
# Laden & afgeleide features
# ---------------------------------------------------------------------------

def load_results(csv_path: str = RESULT_CSV) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Result file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"[INFO] Ingeladen: {csv_path} met shape {df.shape}")
    if "scenario" not in df.columns:
        raise ValueError("Kolom 'scenario' ontbreekt in de resultaten.")
    print(f"[INFO] Scenario's gevonden: {df['scenario'].unique()}")
    return df


def add_basic_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Voegt wat standaard features toe die we misschien gemist hebben in de sim,
    zoals log-ratio's (handig voor plots) en slopes.
    """
    df = df.copy()

    # Log ratio's werken vaak beter in grafieken als de schalen ver uit elkaar liggen
    eps = 1e-6
    if "atk_mean_rate" in df.columns and "base_mean_rate" in df.columns:
        df["log_atk_mean_rate_ratio"] = np.log10(
            (df["atk_mean_rate"] + eps) / (df["base_mean_rate"] + eps)
        )
    if "mit_mean_rate" in df.columns and "base_mean_rate" in df.columns:
        df["log_mit_mean_rate_ratio"] = np.log10(
            (df["mit_mean_rate"] + eps) / (df["base_mean_rate"] + eps)
        )

    # Slopes initialiseren als ze er niet zijn
    for col in ["base_slope", "atk_slope", "mit_slope"]:
        if col not in df.columns:
            df[col] = np.nan

    return df


def add_deepseek_style_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Hier maken we wat complexere features. Het idee is om patronen te vinden
    die niet direct zichtbaar zijn in de ruwe rates (zoals stabiliteit of complexiteit).
    """
    df = df.copy()

    # Hoe stabiel is de rate t.o.v. de baseline?
    if {"base_cv_rate", "atk_cv_rate"}.issubset(df.columns):
        df["rate_stability"] = df["base_cv_rate"] / (df["atk_cv_rate"] + 1e-9)

    # Hoeveel verandert de chaos (entropie)?
    if {"atk_entropy", "base_entropy"}.issubset(df.columns):
        df["entropy_change_ratio"] = (
            df["atk_entropy"] - df["base_entropy"]
        ) / (df["base_entropy"] + 1e-9)

    # Hoe goed herstelt het netwerk zich?
    if {"recovery_rate_drop", "delta_atk_mean_rate"}.issubset(df.columns):
        df["recovery_efficiency"] = df["recovery_rate_drop"] / (
            np.abs(df["delta_atk_mean_rate"]) + 1e-9
        )

    # Combinatie van entropie en burstiness
    if {"atk_entropy", "atk_burstiness"}.issubset(df.columns):
        df["pattern_complexity"] = df["atk_entropy"] * df["atk_burstiness"]

    # Veerkracht maatstaf
    if {"mit_mean_rate", "base_mean_rate", "atk_mean_rate"}.issubset(df.columns):
        df["network_resilience"] = (
            df["mit_mean_rate"] - df["base_mean_rate"]
        ) / (df["atk_mean_rate"] - df["base_mean_rate"] + 1e-9)

    if "delta_atk_mean_rate" in df.columns:
        df["attack_magnitude"] = np.abs(df["delta_atk_mean_rate"])

    # Verschil tussen begin en eind van de aanval (voor trage attacks zoals Drift)
    if {"atk_late_mean_rate", "atk_early_mean_rate"}.issubset(df.columns):
        df["attack_duration_impact"] = (
            df["atk_late_mean_rate"] - df["atk_early_mean_rate"]
        )

    # Log transforms voor features met "long tails"
    for col in ["atk_burstiness", "atk_synchrony", "atk_heterogeneity"]:
        if col in df.columns:
            df[f"log_{col}"] = np.log1p(df[col])

    if {"atk_mean_rate", "base_mean_rate"}.issubset(df.columns):
        df["atk_base_ratio"] = df["atk_mean_rate"] / (df["base_mean_rate"] + 1e-9)

    return df


def ensure_detection_and_risk(df: pd.DataFrame) -> pd.DataFrame:
    """
    Checken of 'detection_delay' en 'clinical_risk' in de data zitten.
    Zo niet, dan leiden we ze hier "quick & dirty" af zodat de plots niet crashen.
    """
    df = df.copy()

    # Delay berekenen uit de absolute tijdstempels
    if "detection_delay" not in df.columns:
        if "first_anomaly_time" in df.columns:
            det = df["first_anomaly_time"] - BASELINE_DURATION_SEC
            det = det.clip(lower=0.0)
            df["detection_delay"] = det
            print("[DERIVED] detection_delay afgeleid uit first_anomaly_time.")
        else:
            df["detection_delay"] = np.nan
            print("[DERIVED] Geen first_anomaly_time; detection_delay blijft NaN.")

    # Risico inschatting (dummy modelletje)
    if "clinical_risk" not in df.columns:
        base_risk_map = {
            "overstimulation": 0.8,
            "poisoning": 0.6,
            "drift": 0.9,
            "svce": 0.7,
            "baseline": 0.0,
        }

        max_delay = max(1.0, np.nanmax(df["detection_delay"].values) if df["detection_delay"].notna().any() else 1.0)

        def _risk_row(row):
            scen = row.get("scenario", "baseline")
            delay = row.get("detection_delay", np.nan)
            if scen == "baseline" or np.isnan(delay):
                return 0.0
            base = base_risk_map.get(scen, 0.5)
            # Normaliseren naar een schaal van 0-10
            norm_delay = min(delay / max_delay, 1.0)
            return float(base * norm_delay * 10.0)

        df["clinical_risk"] = df.apply(_risk_row, axis=1)
        print("[DERIVED] clinical_risk afgeleid uit scenario + detection_delay.")

    return df


# ---------------------------------------------------------------------------
# N-advies & summary / z-scores / effect sizes
# ---------------------------------------------------------------------------

def print_n_advice(df: pd.DataFrame):
    """
    Even checken of we wel genoeg data hebben om serieus genomen te worden.
    """
    if "trial_id" not in df.columns:
        print("[N-ADVICE] Kolom 'trial_id' ontbreekt; n per scenario niet te bepalen.")
        return

    counts = df.groupby("scenario")["trial_id"].nunique()
    print("\n[N-ADVICE] Trials per scenario (unieke trial_id):")
    for scen, c in counts.items():
        print(f"  {scen}: n = {c}")

    min_n = counts.min()
    print(f"\n[N-ADVICE] Kleinste n over scenario's = {min_n}")

    if min_n < 5:
        print("[N-ADVICE] --> Dataset is erg klein. Probeer n=5 of hoger voor betrouwbaarheid.")
    elif min_n < 10:
        print("[N-ADVICE] --> Dit is bruikbaar, maar n=10 is beter voor publicaties.")
    else:
        print("[N-ADVICE] --> Mooi: n is groot genoeg voor goede statistiek.")


def compute_summary_stats(df: pd.DataFrame, out_csv=SUMMARY_CSV):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    summary = df.groupby("scenario")[numeric_cols].agg(["mean", "std"])
    summary.to_csv(out_csv)
    print(f"[SUMMARY] Statistiek per scenario weggeschreven naar: {out_csv}")

    # Nette weergave voor in de console
    nice = pd.DataFrame(index=summary.index)
    for col in numeric_cols:
        m = summary[(col, "mean")]
        s = summary[(col, "std")]
        nice[col] = (m.round(3).astype(str) + " ± " + s.round(3).astype(str))

    print("\n[SUMMARY] Statistiek per scenario (mean ± std):")
    print(nice[numeric_cols])


def compute_zscores_vs_baseline(df: pd.DataFrame, out_csv=ZSCORES_CSV):
    """
    Hoe 'afwijkend' is een aanval t.o.v. normaal gedrag (in standaarddeviaties)?
    """
    baseline = df[df["scenario"] == "baseline"]
    if baseline.empty:
        print("[Z] Geen baseline-rijen gevonden, sla z-scores over.")
        return

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    base_means = baseline[numeric_cols].mean()
    base_stds = baseline[numeric_cols].std().replace(0, np.nan)

    rows = []
    for scen in df["scenario"].unique():
        sub = df[df["scenario"] == scen]
        means = sub[numeric_cols].mean()
        z = (means - base_means) / base_stds
        row = {"scenario": scen}
        row.update(z.to_dict())
        rows.append(row)

    z_df = pd.DataFrame(rows)
    z_df.to_csv(out_csv, index=False)
    print(f"[Z] Z-scores t.o.v. baseline weggeschreven naar: {out_csv}")


def compute_effect_sizes_vs_baseline(df: pd.DataFrame, out_csv=EFFECT_SIZES_CSV):
    """
    Cohen's d berekenen. Vertelt ons of het verschil groot is,
    ongeacht hoeveel data we hebben (in tegenstelling tot p-waardes).
    """
    baseline = df[df["scenario"] == "baseline"]
    if baseline.empty:
        print("[EFFECT SIZES] Geen baseline-rijen, sla Cohen's d over.")
        return

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    results = []

    for scen in df["scenario"].unique():
        if scen == "baseline":
            continue
        sub = df[df["scenario"] == scen]
        row = {"scenario": scen}
        for col in numeric_cols:
            x = sub[col].dropna().values
            y = baseline[col].dropna().values
            if len(x) < 2 or len(y) < 2:
                row[col] = np.nan
                continue
            mean_x, mean_y = x.mean(), y.mean()
            var_x, var_y = x.var(ddof=1), y.var(ddof=1)
            # Pooled standard deviation
            pooled = np.sqrt(
                ((len(x) - 1) * var_x + (len(y) - 1) * var_y) / (len(x) + len(y) - 2)
            )
            if pooled == 0 or np.isnan(pooled):
                row[col] = np.nan
            else:
                row[col] = (mean_x - mean_y) / pooled
        results.append(row)

    eff_df = pd.DataFrame(results)
    eff_df.to_csv(out_csv, index=False)
    print(f"[EFFECT SIZES] Cohen's d t.o.v. baseline weggeschreven naar: {out_csv}")


# ---------------------------------------------------------------------------
# Klassieke figuren
# ---------------------------------------------------------------------------

def _boxplot_feature(df, feature, out_name, out_dir=FIG_DIR):
    """ Helper voor boxplots """
    scenarios = df["scenario"].unique()
    data_to_plot = [df[df["scenario"] == s][feature].dropna() for s in scenarios]
    if all(len(x) == 0 for x in data_to_plot):
        return
    plt.figure()
    plt.boxplot(data_to_plot, tick_labels=scenarios, showmeans=True)
    plt.title(feature)
    plt.xlabel("Scenario")
    plt.ylabel(feature)
    out_path = os.path.join(out_dir, out_name)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[FIG] Boxplot opgeslagen: {out_path}")


def make_classic_boxplots(df: pd.DataFrame):
    # Lijstje van alle plots die we sowieso willen zien
    mapping = [
        ("atk_mean_rate", "box_atk_mean_rate.png"),
        ("atk_burstiness", "box_atk_burstiness.png"),
        ("atk_entropy", "box_atk_entropy.png"),
        ("atk_synchrony", "box_atk_synchrony.png"),
        ("delta_atk_mean_rate", "box_delta_atk_mean_rate.png"),
        ("delta_mit_mean_rate", "box_delta_mit_mean_rate.png"),
        ("pct_atk_mean_rate_change", "box_pct_atk_mean_rate_change.png"),
        ("pct_mit_mean_rate_change", "box_pct_mit_mean_rate_change.png"),
        ("base_heterogeneity", "box_base_heterogeneity.png"),
        ("atk_heterogeneity", "box_atk_heterogeneity.png"),
        ("mit_heterogeneity", "box_mit_heterogeneity.png"),
        ("log_atk_mean_rate_ratio", "box_log_atk_mean_rate_ratio.png"),
        ("log_mit_mean_rate_ratio", "box_log_mit_mean_rate_ratio.png"),
        ("base_slope", "box_base_slope.png"),
        ("atk_slope", "box_atk_slope.png"),
        ("mit_slope", "box_mit_slope.png"),
        ("detection_delay", "box_detection_delay.png"),
        ("clinical_risk", "box_clinical_risk.png"),
    ]
    for col, fname in mapping:
        if col not in df.columns:
            print(f"[WARN] Kolom {col} niet gevonden, sla boxplot over.")
            continue
        _boxplot_feature(df, col, fname)


def make_attack_scatter(df: pd.DataFrame):
    # Scatterplot om clusters te spotten (Burstiness vs Entropy)
    if not {"atk_burstiness", "atk_entropy"}.issubset(df.columns):
        print("[WARN] Vereiste kolommen voor scatterplot ontbreken.")
        return
    plt.figure()
    for scen in df["scenario"].unique():
        sub = df[df["scenario"] == scen]
        plt.scatter(sub["atk_burstiness"], sub["atk_entropy"], label=scen)
    plt.xlabel("Attack burstiness")
    plt.ylabel("Attack entropy")
    plt.legend()
    out_path = os.path.join(FIG_DIR, "scatter_attack_burstiness_vs_entropy.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[FIG] Attack-scatterplot opgeslagen: {out_path}")


def make_svce_barplot(df: pd.DataFrame):
    # Specifiek voor het virus-scenario: hoe verdeeld is S-E-I-R?
    cols = ["svce_frac_S", "svce_frac_E", "svce_frac_I", "svce_frac_R"]
    if not all(c in df.columns for c in cols):
        return
    sub = df[df["scenario"] == "svce"]
    if sub.empty:
        return
    means = sub[cols].mean()
    plt.figure()
    plt.bar(cols, means.values)
    plt.ylabel("Fraction")
    plt.title("sVCE SEIR-fractions (scenario 'svce')")
    out_path = os.path.join(FIG_DIR, "bar_svce_seir_fractions.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[FIG] sVCE SEIR-barplot opgeslagen: {out_path}")


def make_bland_altman_plot(df: pd.DataFrame):
    # Klinische validatie plot: agreement tussen baseline en attack
    if not {"base_mean_rate", "atk_mean_rate"}.issubset(df.columns):
        return
    x = df["base_mean_rate"].values
    y = df["atk_mean_rate"].values
    mean = (x + y) / 2
    diff = y - x
    md = np.mean(diff)
    sd = np.std(diff, ddof=1)

    plt.figure()
    plt.scatter(mean, diff, alpha=0.7)
    plt.axhline(md, color="red", linestyle="--", label="Mean diff")
    plt.axhline(md + 1.96 * sd, color="gray", linestyle=":")
    plt.axhline(md - 1.96 * sd, color="gray", linestyle=":")
    plt.xlabel("Mean of base and attack mean_rate")
    plt.ylabel("Attack - Base mean_rate")
    plt.title("Bland–Altman: base vs attack mean_rate")
    plt.legend()
    out_path = os.path.join(FIG_DIR, "bland_altman_base_vs_attack_mean_rate.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[FIG] Bland–Altman plot opgeslagen: {out_path}")


def make_zscore_heatmap(zscore_csv=ZSCORES_CSV):
    if not os.path.exists(zscore_csv):
        print("[FIG] Geen Z-score CSV gevonden, sla heatmap over.")
        return
    z = pd.read_csv(zscore_csv)
    if "scenario" not in z.columns:
        print("[FIG] Z-score CSV bevat geen 'scenario'-kolom.")
        return
    z = z.set_index("scenario")
    plt.figure(figsize=(8, 6))
    sns.heatmap(z, cmap="coolwarm", center=0, annot=False)
    plt.title("Z-score heatmap vs baseline")
    out_path = os.path.join(FIG_DIR, "zscore_heatmap_scenarios.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[FIG] Z-score heatmap opgeslagen: {out_path}")


# ---------------------------------------------------------------------------
# Deepseek-style visualisaties (correlatie, PCA, detection)
# ---------------------------------------------------------------------------

_scaler = StandardScaler()
_label_encoder = LabelEncoder()


def create_correlation_heatmap(df: pd.DataFrame, save_dir=ENH_FIG_DIR):
    """ Welke features correleren sterk met elkaar (of met detection_delay)? """
    numeric_df = df.select_dtypes(include=[np.number])
    if len(numeric_df.columns) < 2:
        return

    corr_matrix = numeric_df.corr()

    if "detection_delay" in corr_matrix.columns:
        corr_with_target = corr_matrix["detection_delay"].abs().sort_values(
            ascending=False
        ).index[:12]
    else:
        ref_col = numeric_df.columns[0]
        corr_with_target = corr_matrix[ref_col].abs().sort_values(
            ascending=False
        ).index[:12]

    top_corr = corr_matrix.loc[corr_with_target, corr_with_target]

    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(top_corr, dtype=bool))
    sns.heatmap(
        top_corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        square=True,
        cbar_kws={"shrink": 0.8},
    )
    plt.title("Enhanced Feature Correlation Heatmap", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out_path = os.path.join(save_dir, "correlation_heatmap.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[FIG] Correlation heatmap opgeslagen: {out_path}")


def create_scenario_comparison_plots(df: pd.DataFrame, save_dir=ENH_FIG_DIR):
    # Mooiere boxplots en violins voor de belangrijkste features
    key_features = [
        "atk_mean_rate",
        "atk_burstiness",
        "atk_entropy",
        "atk_heterogeneity",
    ]
    key_features = [f for f in key_features if f in df.columns]

    for feature in key_features:
        plt.figure(figsize=(9, 6))
        sns.boxplot(data=df, x="scenario", y=feature)
        plt.title(f"{feature} by Scenario", fontweight="bold")
        plt.xticks(rotation=45)
        plt.tight_layout()
        out_path = os.path.join(save_dir, f"boxplot_{feature}.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[FIG] Enhanced boxplot opgeslagen: {out_path}")

    # Violin plots geven beter inzicht in de distributie (bv. bimodale data)
    for feature in key_features[:2]:
        plt.figure(figsize=(9, 6))
        sns.violinplot(data=df, x="scenario", y=feature)
        plt.title(f"Distribution of {feature} by Scenario", fontweight="bold")
        plt.xticks(rotation=45)
        plt.tight_layout()
        out_path = os.path.join(save_dir, f"violin_{feature}.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[FIG] Violin-plot opgeslagen: {out_path}")


def create_temporal_profile_plot(df: pd.DataFrame, save_dir=ENH_FIG_DIR):
    """ Laat het verloop in de tijd zien (Baseline -> Attack -> Mitigation) """
    temporal_features = [
        "base_mean_rate",
        "atk_early_mean_rate",
        "atk_late_mean_rate",
        "mit_early_mean_rate",
        "mit_mean_rate",
    ]
    temporal_features = [f for f in temporal_features if f in df.columns]
    if len(temporal_features) < 3:
        return

    plt.figure(figsize=(10, 7))
    for scen in df["scenario"].unique():
        sub = df[df["scenario"] == scen]
        means = sub[temporal_features].mean()
        stds = sub[temporal_features].std()
        x_pos = np.arange(len(temporal_features))
        plt.plot(x_pos, means, "o-", linewidth=2, label=scen, alpha=0.8)
        plt.fill_between(x_pos, means - stds, means + stds, alpha=0.2)

    pretty_labels = [
        f.replace("_mean_rate", "")
         .replace("_", " ")
         .title()
        for f in temporal_features
    ]
    plt.xticks(np.arange(len(temporal_features)), pretty_labels, rotation=30)
    plt.xlabel("Time phase")
    plt.ylabel("Mean rate (Hz)")
    plt.title("Temporal Profile Across Scenarios", fontweight="bold")
    plt.legend()
    plt.grid(True, alpha=0.3)
    out_path = os.path.join(save_dir, "temporal_profile.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[FIG] Temporal profile opgeslagen: {out_path}")


def create_pca_tsne_plot(df: pd.DataFrame, save_dir=ENH_FIG_DIR):
    """ Dimensiereductie: kunnen we de scenario's uit elkaar houden in 2D? """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    feature_cols = [
        col
        for col in numeric_cols
        if col not in ["trial_id", "first_anomaly_time"]
    ]
    if len(feature_cols) < 2 or "scenario" not in df.columns:
        return

    X = df[feature_cols].fillna(0.0)
    y = df["scenario"]
    X_scaled = _scaler.fit_transform(X)

    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    plt.figure(figsize=(9, 7))
    scatter = plt.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=_label_encoder.fit_transform(y),
        cmap="viridis",
        alpha=0.7,
        s=60,
    )
    plt.colorbar(scatter, label="Scenario")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)")
    plt.title("PCA – Scenario Separation", fontweight="bold")
    plt.grid(True, alpha=0.3)
    out_path = os.path.join(save_dir, "pca_plot.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[FIG] PCA plot opgeslagen: {out_path}")

    # t-SNE (alleen doen als we genoeg punten hebben, anders is het ruis)
    if len(X) > 10:
        try:
            tsne = TSNE(
                n_components=2,
                random_state=42,
                perplexity=min(5, len(X) - 1),
            )
            X_tsne = tsne.fit_transform(X_scaled)
            plt.figure(figsize=(9, 7))
            scatter = plt.scatter(
                X_tsne[:, 0],
                X_tsne[:, 1],
                c=_label_encoder.transform(y),
                cmap="viridis",
                alpha=0.7,
                s=60,
            )
            plt.colorbar(scatter, label="Scenario")
            plt.xlabel("t-SNE 1")
            plt.ylabel("t-SNE 2")
            plt.title("t-SNE – Scenario Separation", fontweight="bold")
            plt.grid(True, alpha=0.3)
            out_path = os.path.join(save_dir, "tsne_plot.png")
            plt.tight_layout()
            plt.savefig(out_path, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"[FIG] t-SNE plot opgeslagen: {out_path}")
        except Exception as e:
            print(f"[WARN] t-SNE mislukt: {e}")


def create_detection_analysis_plots(df: pd.DataFrame, save_dir=ENH_FIG_DIR):
    if "detection_delay" not in df.columns:
        print("[DETECTION] Geen detection_delay-kolom, sla detection-plots over.")
        return

    attack_df = df[df["scenario"] != "baseline"]
    if attack_df.empty:
        return

    det_data = attack_df.dropna(subset=["detection_delay"])

    if not det_data.empty:
        plt.figure(figsize=(9, 6))
        sns.boxplot(data=det_data, x="scenario", y="detection_delay")
        plt.title("Detection delay by attack scenario", fontweight="bold")
        plt.ylabel("Detection delay (s)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        out_path = os.path.join(save_dir, "detection_delay.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[FIG] Detection-delay plot opgeslagen: {out_path}")

    # Clinical risk plotten
    if "clinical_risk" in df.columns:
        risk_data = attack_df.dropna(subset=["clinical_risk"])

        if not risk_data.empty:
            plt.figure(figsize=(9, 6))
            sns.boxplot(data=risk_data, x="scenario", y="clinical_risk")
            plt.title("Clinical risk by attack scenario", fontweight="bold")
            plt.ylabel("Clinical risk score")
            plt.xticks(rotation=45)
            plt.tight_layout()
            out_path = os.path.join(save_dir, "clinical_risk.png")
            plt.savefig(out_path, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"[FIG] Clinical-risk plot opgeslagen: {out_path}")


# ---------------------------------------------------------------------------
# ML: multi-class + binary + anomaly detection
# ---------------------------------------------------------------------------

def build_and_evaluate_classifier(df: pd.DataFrame, out_report=CLASSIFIER_REPORT_TXT):
    """
    Kan een Machine Learning model (Random Forest) automatisch het scenario herkennen
    op basis van de metrics?
    """
    lines = []
    lines.append("=== Living Crime Scene – Classifier Report ===\n")

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    feature_cols = [
        col
        for col in numeric_cols
        if col not in ["trial_id", "first_anomaly_time"]
    ]
    if not feature_cols:
        lines.append("[SKIP] Geen numerieke features gevonden.\n")
        with open(out_report, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"[ML] Classifier-rapport weggeschreven naar: {out_report}")
        return

    X = df[feature_cols].fillna(0.0).values
    y = df["scenario"].values

    n_samples = X.shape[0]
    unique_scenarios, counts = np.unique(y, return_counts=True)
    min_per_class = counts.min()

    lines.append(f"Features: {len(feature_cols)}\n")
    lines.append(f"Samples: {n_samples}\n")
    lines.append("Class distribution:\n")
    for c, cnt in zip(unique_scenarios, counts):
        lines.append(f"  {c}: n={cnt}")
    lines.append("")

    # Multi-class Classifier
    if n_samples >= len(unique_scenarios) * 2 and min_per_class >= 2:
        test_size = max(0.3, len(unique_scenarios) / n_samples)
        test_size = min(0.5, test_size)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=42
        )

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        clf = RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced",
        )
        clf.fit(X_train_s, y_train)
        y_pred = clf.predict(X_test_s)

        acc = accuracy_score(y_test, y_pred)
        lines.append(f"[MULTI-CLASS] Hold-out accuracy: {acc:.3f}\n")
        lines.append("Classification report (hold-out):\n")
        lines.append(classification_report(y_test, y_pred))
        cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
        lines.append("Confusion matrix (hold-out):\n")
        lines.append(str(cm))
        lines.append("")

        # Feature importance: welke features 'verraden' de aanval het meest?
        feat_imp = pd.DataFrame(
            {"feature": feature_cols, "importance": clf.feature_importances_}
        ).sort_values("importance", ascending=False)
        lines.append("Top 10 features (multi-class RF):\n")
        for _, row in feat_imp.head(10).iterrows():
            lines.append(f"  {row['feature']}: {row['importance']:.4f}")
        lines.append("")

        # Cross-validation (robuuster dan hold-out)
        if min_per_class >= 2:
            k = min(5, min_per_class)
            scaler_full = StandardScaler()
            X_scaled = scaler_full.fit_transform(X)
            cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
            cv_scores = cross_val_score(clf, X_scaled, y, cv=cv)
            lines.append(
                f"[MULTI-CLASS] Stratified {k}-fold CV accuracy: "
                f"{cv_scores.mean():.3f} ± {cv_scores.std():.3f}\n"
            )
    else:
        lines.append(
            "[MULTI-CLASS] Dataset te klein voor betrouwbare hold-out / CV.\n"
        )

    # Binary Classifier: Baseline vs The Rest
    df_bin = df.copy()
    df_bin["is_attack"] = (df_bin["scenario"] != "baseline").astype(int)
    if df_bin["is_attack"].nunique() < 2:
        lines.append("[BINARY] Geen variatie (alle samples baseline of alle attack).\n")
    else:
        Xb = df_bin[feature_cols].fillna(0.0).values
        yb = df_bin["is_attack"].values
        unique_bin, counts_bin = np.unique(yb, return_counts=True)
        min_per_class_bin = counts_bin.min()
        lines.append(
            f"[BINARY] Samples: {len(Xb)}, min_per_class: {min_per_class_bin}\n"
        )

        if min_per_class_bin >= 2:
            k_bin = min(5, min_per_class_bin)
            scaler_b = StandardScaler()
            Xb_scaled = scaler_b.fit_transform(Xb)

            clf_bin = RandomForestClassifier(
                n_estimators=150,
                random_state=42,
                class_weight="balanced",
            )
            cvb = StratifiedKFold(n_splits=k_bin, shuffle=True, random_state=42)
            cv_scores_bin = cross_val_score(clf_bin, Xb_scaled, yb, cv=cvb)
            lines.append(
                f"[BINARY] Stratified {k_bin}-fold CV accuracy: "
                f"{cv_scores_bin.mean():.3f} ± {cv_scores_bin.std():.3f}\n"
            )
        else:
            lines.append(
                "[BINARY] Te weinig samples per klasse voor binary cross-val.\n"
            )

    with open(out_report, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[ML] Classifier-rapport weggeschreven naar: {out_report}")


def anomaly_detection_analysis(df: pd.DataFrame):
    """
    Unsupervised detection: trainen op baseline en kijken of we de rest
    als 'abnormaal' kunnen markeren (IsolationForest).
    """
    if "scenario" not in df.columns:
        return
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    feature_cols = [
        col
        for col in numeric_cols
        if col not in ["trial_id", "first_anomaly_time", "detection_delay", "clinical_risk"]
    ]
    if len(feature_cols) < 2:
        print("[UNSUPERVISED] Te weinig numerieke features voor IsolationForest.")
        return

    X = df[feature_cols].fillna(0.0).values
    baseline_mask = df["scenario"] == "baseline"
    if baseline_mask.sum() < 6:
        print("[UNSUPERVISED] Te weinig baseline-samples voor IsolationForest, sla over.")
        return

    # Train alleen op normaal gedrag
    X_baseline = X[baseline_mask]
    scaler = StandardScaler()
    Xb_scaled = scaler.fit_transform(X_baseline)

    iso = IsolationForest(contamination=0.1, random_state=42)
    iso.fit(Xb_scaled)

    # Test op alles
    X_all_scaled = scaler.transform(X)
    anomalies = iso.predict(X_all_scaled)
    anomaly_labels = (anomalies == -1).astype(int) # -1 is anomaly

    true_attacks = (df["scenario"] != "baseline").astype(int)
    acc = accuracy_score(true_attacks, anomaly_labels)
    print("\n[UNSUPERVISED] IsolationForest anomaly detection:")
    print(f"  Accuracy vs. ground truth attack/non-attack: {acc:.3f}")
    cm = confusion_matrix(true_attacks, anomaly_labels)
    print("  Confusion matrix (true vs predicted anomaly):")
    print(cm)


# ---------------------------------------------------------------------------
# Statistische analyse (ANOVA/T-tests, indien SciPy aanwezig)
# ---------------------------------------------------------------------------

def enhanced_statistical_analysis(df: pd.DataFrame):
    if not SCIPY_AVAILABLE:
        print("[STATS] SciPy niet beschikbaar, sla ANOVA/post-hoc over.")
        return {}

    print("\n============================================")
    print("STATISTICAL ANALYSIS (ANOVA + post-hoc)")
    print("============================================")

    analysis_features = [
        "atk_mean_rate",
        "atk_burstiness",
        "atk_entropy",
        "atk_heterogeneity",
        "detection_delay",
    ]
    analysis_features = [f for f in analysis_features if f in df.columns]

    results = {}
    for feature in analysis_features:
        print(f"\n--- Analysis for {feature} ---")
        groups = []
        names = []
        for scen in df["scenario"].unique():
            vals = df[df["scenario"] == scen][feature].dropna().values
            if len(vals) > 0:
                groups.append(vals)
                names.append(scen)

        if len(groups) < 2:
            print("  Niet genoeg groepen voor ANOVA.")
            continue

        try:
            # ANOVA
            f_stat, p_val = stats.f_oneway(*groups)
            print(f"  ANOVA: F={f_stat:.4f}, p={p_val:.4f}")
            if p_val < 0.05:
                print("  Significante verschillen tussen scenario's.")
                from itertools import combinations
                print("  Post-hoc t-tests (Welch, ongepaard):")
                # Even alle combinaties testen (Welch's t-test)
                for i, j in combinations(range(len(groups)), 2):
                    if len(groups[i]) > 1 and len(groups[j]) > 1:
                        t_stat, p_t = stats.ttest_ind(
                            groups[i], groups[j], equal_var=False
                        )
                        if p_t < 0.001:
                            sig = "***"
                        elif p_t < 0.01:
                            sig = "**"
                        elif p_t < 0.05:
                            sig = "*"
                        else:
                            sig = "ns"
                        print(
                            f"    {names[i]} vs {names[j]}: p={p_t:.4f} ({sig})"
                        )
            else:
                print("  Geen significante verschillen tussen scenario's.")
            results[feature] = {"f_stat": f_stat, "p_value": p_val}
        except Exception as e:
            print(f"  ANOVA mislukt: {e}")

    return results


# ---------------------------------------------------------------------------
# Korte checklist voor "is deze run bruikbaar?"
# ---------------------------------------------------------------------------

def print_quick_quality_check(df: pd.DataFrame):
    """
    Laatste check voordat je de resultaten naar je baas/klant stuurt.
    """
    print("\n============================================")
    print("QUICK QUALITY CHECK")
    print("============================================")

    # 1. n per scenario
    if "trial_id" in df.columns:
        counts = df.groupby("scenario")["trial_id"].nunique()
        print("1) n per scenario:")
        print(counts)
        if counts.min() < 5:
            print("   -> Minstens één scenario heeft n<5 (kwetsbaar).")
        elif counts.min() < 10:
            print("   -> n~5–9: bruikbaar, maar niet super robuust.")
        else:
            print("   -> n>=10 overal: mooi.")

    # 2. variatie in kernfeatures
    core_cols = [
        "atk_mean_rate",
        "atk_burstiness",
        "atk_entropy",
        "atk_heterogeneity",
    ]
    core_cols = [c for c in core_cols if c in df.columns]
    if core_cols:
        print("\n2) Variatie (std) in kernfeatures:")
        for col in core_cols:
            s = df[col].std()
            print(f"   {col}: std={s:.3f}")
        print("   -> Als alles ~0 is, zijn scenario's nauwelijks te onderscheiden.")

    # 3. detection_delay / clinical_risk aanwezig?
    print("\n3) detection_delay / clinical_risk:")
    for col in ["detection_delay", "clinical_risk"]:
        if col in df.columns and df[col].notna().any():
            print(f"   {col}: OK (min={df[col].min():.3f}, max={df[col].max():.3f})")
        else:
            print(f"   {col}: ontbreekt of overal NaN.")


# ---------------------------------------------------------------------------
# Compact tekstueel eindrapport
# ---------------------------------------------------------------------------

def generate_comprehensive_report(df: pd.DataFrame, feature_importance=None,
                                  out_path=ENHANCED_REPORT_TXT):
    report = []
    report.append("ENHANCED LIVING CRIME SCENE ANALYSIS REPORT")
    report.append("=" * 60)
    report.append(f"Total samples: {len(df)}")
    report.append(f"Scenarios: {', '.join(df['scenario'].unique())}")
    report.append("")

    scen_summary = df.groupby("scenario").size()
    report.append("Scenario distribution:")
    for scen, cnt in scen_summary.items():
        report.append(f"  {scen}: {cnt} samples")
    report.append("")

    key_metrics = ["atk_mean_rate", "atk_burstiness", "detection_delay"]
    key_metrics = [m for m in key_metrics if m in df.columns]

    if key_metrics:
        report.append("Key metrics (mean ± std):")
        for metric in key_metrics:
            report.append(f"  {metric}:")
            for scen in df["scenario"].unique():
                vals = df[df["scenario"] == scen][metric].dropna()
                if len(vals) > 0:
                    report.append(
                        f"    {scen}: {vals.mean():.3f} ± {vals.std():.3f}"
                    )
        report.append("")

    if feature_importance is not None:
        report.append("Top 5 most important features (multi-class RF):")
        for _, row in feature_importance.head(5).iterrows():
            report.append(f"  {row['feature']}: {row['importance']:.4f}")
        report.append("")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    print(f"[REPORT] Samenvattend rapport weggeschreven naar: {out_path}")
    return report


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------

def main():
    start = time.time()

    df = load_results(RESULT_CSV)
    df = add_basic_derived_features(df)
    df = add_deepseek_style_features(df)
    df = ensure_detection_and_risk(df)

    # N-advies + summary / z-score / effect sizes
    print_n_advice(df)
    compute_summary_stats(df, SUMMARY_CSV)
    compute_zscores_vs_baseline(df, ZSCORES_CSV)
    compute_effect_sizes_vs_baseline(df, EFFECT_SIZES_CSV)

    # Klassieke figuren maken
    make_classic_boxplots(df)
    make_attack_scatter(df)
    make_svce_barplot(df)
    make_bland_altman_plot(df)
    make_zscore_heatmap(ZSCORES_CSV)

    # Enhaced / deepseek-style figuren maken
    print(f"\n[FIG] Enhanced visualisaties in map: {ENH_FIG_DIR}")
    create_correlation_heatmap(df, ENH_FIG_DIR)
    create_scenario_comparison_plots(df, ENH_FIG_DIR)
    create_temporal_profile_plot(df, ENH_FIG_DIR)
    create_pca_tsne_plot(df, ENH_FIG_DIR)
    create_detection_analysis_plots(df, ENH_FIG_DIR)

    # Machine Learning + anomaly detection
    print("\n[ML] Training classifiers...")
    build_and_evaluate_classifier(df, CLASSIFIER_REPORT_TXT)
    anomaly_detection_analysis(df)

    # Statistiek
    enhanced_statistical_analysis(df)

    # Korte kwaliteitscheck
    print_quick_quality_check(df)

    generate_comprehensive_report(df, feature_importance=None,
                                  out_path=ENHANCED_REPORT_TXT)

    elapsed = time.time() - start
    print(f"\n[DONE] Analyse afgerond in {timedelta(seconds=int(elapsed))}.")


if __name__ == "__main__":
    main()