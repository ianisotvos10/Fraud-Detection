import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import average_precision_score

from src.config import DROP_COLUMNS, REPORTS_DIR, TEST_PATH, VAL_PATH
from src.drift import (
    MIN_DRIFTED_FEATURES,
    PSI_LIMIT,
    SHIFTED_FEATURES,
    build_batches,
    get_top_features,
    psi,
)
from src.tracking import load_serving_model, load_threshold


def plot_report(report, ref_flag_rate, scenario, path):
    alerted = report.loc[report["alert"], "batch"].tolist()
    labels = [f"{int(b)}\nn={int(f)}" for b, f in zip(report["batch"], report["frauds"])]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    ax = axes[0]
    ax.plot(report["batch"], np.clip(report["score_psi"], 1e-3, None), marker="o", label="PSI scor")
    ax.plot(report["batch"], np.clip(report["max_feature_psi"], 1e-3, None), marker="s", label="PSI maxim feature")
    ax.axhline(PSI_LIMIT, linestyle="--", color="red", label="prag alerta")
    ax.set_yscale("log")
    ax.set_ylim(1e-3, 10)
    ax.set_ylabel("PSI")
    ax.set_title("Drift pe loturi (scara logaritmica)")

    ax = axes[1]
    ax.plot(report["batch"], report["recall"], marker="o", label="Recall")
    ax.plot(report["batch"], report["precision"], marker="s", label="Precizie")
    ax.set_ylim(0, 1.05)
    ax.set_title("Performanta pe loturi")

    ax = axes[2]
    ax.plot(report["batch"], report["flag_rate"] * 100, marker="o", label="Tranzactii marcate")
    ax.axhline(ref_flag_rate * 100, linestyle="--", color="gray", label="referinta (validare)")
    ax.set_ylim(bottom=0)
    ax.set_ylabel("%")
    ax.set_title("Procent tranzactii marcate ca frauda")

    for ax in axes:
        ax.set_xticks(report["batch"])
        ax.set_xticklabels(labels)
        ax.set_xlabel("Lot (n = numar fraude)")
        for b in alerted:
            ax.axvspan(b - 0.5, b + 0.5, color="red", alpha=0.12)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.4), ncol=2, fontsize=8)

    fig.suptitle(f"Scenariu: {scenario} (banda rosie = lot cu alerta)")
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def run(scenario):
    print("Incarcam modelul si datele de referinta...")
    model, run_id = load_serving_model()
    threshold = load_threshold(run_id)

    ref_df = pd.read_csv(VAL_PATH)
    stream_df = pd.read_csv(TEST_PATH).sort_values("Time").reset_index(drop=True)

    X_ref = ref_df.drop(columns=DROP_COLUMNS)
    ref_scores = model.predict_proba(X_ref)[:, 1]
    ref_flag_rate = float((ref_scores >= threshold).mean())
    std = X_ref.std()

    top = get_top_features(model, X_ref.columns)
    batches = build_batches(stream_df, scenario, top[:SHIFTED_FEATURES], std)

    print(f"Scenariu: {scenario}, prag model: {threshold:.4f}")
    print(f"Feature-uri monitorizate: {', '.join(top)}")
    print(f"Procent de referinta tranzactii marcate: {ref_flag_rate * 100:.3f}%")

    rows = []
    for number, X, y in batches:
        scores = model.predict_proba(X)[:, 1]
        pred = scores >= threshold

        feature_psi = {name: psi(X_ref[name].to_numpy(), X[name].to_numpy()) for name in top}
        drifted = sum(value > PSI_LIMIT for value in feature_psi.values())
        score_psi = psi(ref_scores, scores)

        frauds = int(y.sum())
        tp = int((pred & (y == 1)).sum())
        n_pred = int(pred.sum())

        rows.append(
            {
                "batch": number,
                "rows": len(X),
                "frauds": frauds,
                "score_psi": round(score_psi, 4),
                "score_ks": round(float(ks_2samp(ref_scores, scores).statistic), 4),
                "max_feature_psi": round(max(feature_psi.values()), 4),
                "drifted_features": drifted,
                "mean_score": round(float(scores.mean()), 5),
                "flag_rate": round(float(pred.mean()), 5),
                "pr_auc": average_precision_score(y, scores) if frauds > 0 else np.nan,
                "recall": tp / frauds if frauds > 0 else np.nan,
                "precision": tp / n_pred if n_pred > 0 else np.nan,
                "alert": bool(score_psi > PSI_LIMIT or drifted >= MIN_DRIFTED_FEATURES),
            }
        )

    report = pd.DataFrame(rows)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    csv_path = os.path.join(REPORTS_DIR, f"drift_report_{scenario}.csv")
    png_path = os.path.join(REPORTS_DIR, f"drift_{scenario}.png")
    report.to_csv(csv_path, index=False)

    print("\n--- Raport monitorizare ---")
    print(report.round(4).to_string(index=False))

    alerted = report.loc[report["alert"], "batch"].tolist()
    if alerted:
        print(f"\nALERTA: drift detectat in loturile {alerted}")
    else:
        print("\nNu s-a detectat drift in niciun lot.")

    plot_report(report, ref_flag_rate, scenario, png_path)

    print(f"\nRaport salvat in {csv_path}")
    print(f"Grafic salvat in {png_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=["natural", "shift"], default="natural")
    args = parser.parse_args()
    run(args.scenario)
