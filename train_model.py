"""Treinamento auditável do modelo de risco de defasagem.

Principais garantias:
- target explícito e configurável;
- pares longitudinais preservam RA e período;
- avaliação principal temporal (2022->2023 para treino, 2023->2024 para teste);
- threshold escolhido somente dentro do conjunto de treino;
- comparação com baseline de persistência e perfil acionável;
- Pipeline completo salvo no artefato de produção.

Uso:
    python train_model.py --data BASE_DE_DADOS_PEDE_2024_-_DATATHON.xlsx
"""

from __future__ import annotations

import argparse
import json
import platform
import re
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from model_utils import (
    BASE_RAW_FEATURES,
    HARD_LIMITS,
    OPTIONAL_RAW_FEATURES,
    FeatureBuilder,
    predict_positive_probability,
    prepare_numeric_frame,
)


TARGET_MODES = {
    "future_defasagem": "risco = 1 quando Defasagem_N1 <= -1, independentemente do estado em N",
    "new_defasagem": "risco = 1 quando Defasagem_N1 <= -1 entre alunos com Defasagem_N >= 0",
}


def normalized_columns(df: pd.DataFrame) -> dict[str, str]:
    return {str(c).strip().casefold(): c for c in df.columns}


def find_column(df: pd.DataFrame, candidates: list[str], required: bool = True):
    lookup = normalized_columns(df)
    for candidate in candidates:
        key = candidate.strip().casefold()
        if key in lookup:
            return lookup[key]
    if required:
        raise KeyError(f"Nenhuma das colunas foi encontrada: {candidates}")
    return None


def fase_num(value):
    if pd.isna(value):
        return np.nan
    text = str(value).upper().strip()
    if text.startswith("ALFA"):
        return 0
    match = re.search(r"(\d+)", text)
    return int(match.group(1)) if match else np.nan


def load_year(path: str | Path, sheet: str, year: int) -> pd.DataFrame:
    source = pd.read_excel(path, sheet_name=sheet)
    out = pd.DataFrame()
    out["RA"] = source[find_column(source, ["RA"])].astype("string").str.strip()
    out["Ano"] = year
    fase_col = find_column(source, ["Fase", f"Fase {year}"])
    out["Fase_num"] = source[fase_col].apply(fase_num)

    inde_col = find_column(source, [f"INDE {str(year)[-2:]}", f"INDE {year}", "INDE"])
    out["INDE"] = source[inde_col]

    pedra_col = find_column(source, [f"Pedra {str(year)[-2:]}", f"Pedra {year}", "Pedra"], required=False)
    out["Pedra"] = source[pedra_col] if pedra_col else pd.NA
    out["Pedra"] = out["Pedra"].replace({"Agata": "Ágata", "INCLUIR": pd.NA})

    for indicator in ["IAA", "IEG", "IPS", "IPP", "IDA", "IAN", "IPV"]:
        col = find_column(source, [indicator, f"{indicator} {year}", f"{indicator} {str(year)[-2:]}"], required=False)
        out[indicator] = source[col] if col else np.nan

    defas_col = find_column(source, ["Defas", "Defasagem", f"Defas {year}", f"Defasagem {year}"])
    out["Defasagem"] = source[defas_col]

    numeric = ["INDE", "Fase_num", "IAA", "IEG", "IPS", "IPP", "IDA", "IAN", "IPV", "Defasagem"]
    out = prepare_numeric_frame(out, numeric)
    out.loc[out["RA"].isin(["", "<NA>"]), "RA"] = pd.NA
    return out


def validate_year(df: pd.DataFrame, year: int) -> None:
    if df["RA"].isna().any():
        raise ValueError(f"Existem RAs nulos em {year}.")
    duplicated = df.loc[df["RA"].duplicated(keep=False), "RA"].unique().tolist()
    if duplicated:
        sample = duplicated[:10]
        raise ValueError(f"RA deve ser único em {year}. Duplicados (amostra): {sample}")


def longitudinal_pairs(full: pd.DataFrame, year_n: int, year_n1: int) -> pd.DataFrame:
    columns = ["RA", "IAA", "IEG", "IPS", "IPP", "IDA", "IAN", "IPV", "Defasagem", "Fase_num"]
    left = full.loc[full["Ano"] == year_n, columns].copy()
    right = full.loc[full["Ano"] == year_n1, columns].copy()
    validate_year(left, year_n)
    validate_year(right, year_n1)

    left = left.rename(columns={c: f"{c}_N" for c in columns if c != "RA"})
    right = right.rename(columns={c: f"{c}_N1" for c in columns if c != "RA"})
    pairs = left.merge(right, on="RA", how="inner", validate="one_to_one")
    pairs["Ano_N"] = year_n
    pairs["Ano_N1"] = year_n1
    pairs = pairs.rename(columns={"Fase_num_N": "Fase_num", "Fase_num_N1": "Fase_num_N1"})
    return pairs


def build_dataset(path: str | Path, target_mode: str, ipp_min_coverage: float = 0.40):
    if target_mode not in TARGET_MODES:
        raise ValueError(f"target_mode inválido: {target_mode}")

    years = {
        2022: load_year(path, "PEDE2022", 2022),
        2023: load_year(path, "PEDE2023", 2023),
        2024: load_year(path, "PEDE2024", 2024),
    }
    full = pd.concat(years.values(), ignore_index=True)
    pairs = pd.concat(
        [
            longitudinal_pairs(full, 2022, 2023),
            longitudinal_pairs(full, 2023, 2024),
        ],
        ignore_index=True,
    )

    pairs = pairs[pairs["Defasagem_N1"].notna()].copy()
    if target_mode == "new_defasagem":
        pairs = pairs[pairs["Defasagem_N"].ge(0)].copy()
    pairs["risco_defasagem"] = pairs["Defasagem_N1"].le(-1).astype(int)

    raw_features = list(BASE_RAW_FEATURES)
    ipp_coverage = float(pairs["IPP_N"].notna().mean())
    if ipp_coverage >= ipp_min_coverage:
        raw_features += OPTIONAL_RAW_FEATURES

    required = ["RA", "Ano_N", "Ano_N1", "risco_defasagem"] + raw_features
    dataset = pairs[required].copy()
    if dataset["risco_defasagem"].nunique() < 2:
        raise ValueError("O target possui somente uma classe após os filtros.")

    diagnostics = {
        "rows_full": int(len(full)),
        "rows_pairs": int(len(pairs)),
        "ipp_coverage": round(ipp_coverage, 4),
        "positive_rate": round(float(dataset["risco_defasagem"].mean()), 4),
        "transitions": {
            f"{int(row.Ano_N)}-{int(row.Ano_N1)}": int(row.n)
            for row in dataset.groupby(["Ano_N", "Ano_N1"]).size().rename("n").reset_index().itertuples()
        },
    }
    return dataset, full, raw_features, diagnostics


def build_pipeline(raw_features: list[str], profile: str, random_state: int = 42) -> Pipeline:
    return Pipeline(
        [
            ("features", FeatureBuilder(raw_features=raw_features, profile=profile)),
            ("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=500,
                    max_depth=7,
                    min_samples_leaf=4,
                    class_weight="balanced_subsample",
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def build_calibrated_model(raw_features: list[str], profile: str, y: pd.Series):
    min_class = int(y.value_counts().min())
    cv = min(3, min_class)
    base = build_pipeline(raw_features, profile)
    if cv < 2:
        return base, False
    return CalibratedClassifierCV(estimator=base, method="sigmoid", cv=cv), True


def safe_auc(y_true, probabilities):
    return float(roc_auc_score(y_true, probabilities)) if pd.Series(y_true).nunique() == 2 else None


def select_threshold(y_true, probabilities, recall_target: float = 0.80) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    if len(thresholds) == 0:
        return 0.50
    valid = np.where(recall[:-1] >= recall_target)[0]
    if len(valid) == 0:
        return 0.50
    best = valid[np.argmax(precision[:-1][valid])]
    return float(np.clip(thresholds[best], 0.05, 0.95))


def classification_metrics(y_true, probabilities, threshold: float) -> dict:
    y_pred = (np.asarray(probabilities) >= threshold).astype(int)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "roc_auc": None if safe_auc(y_true, probabilities) is None else round(safe_auc(y_true, probabilities), 4),
        "pr_auc": round(float(average_precision_score(y_true, probabilities)), 4),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "brier": round(float(brier_score_loss(y_true, probabilities)), 4),
        "threshold": round(float(threshold), 4),
        "confusion_matrix": matrix.astype(int).tolist(),
        "n": int(len(y_true)),
        "positive_rate": round(float(np.mean(y_true)), 4),
    }


def inner_threshold(X_train, y_train, raw_features, profile, recall_target, random_state=42):
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.25,
        random_state=random_state,
        stratify=y_train,
    )
    model, _ = build_calibrated_model(raw_features, profile, y_fit)
    model.fit(X_fit, y_fit)
    probabilities = predict_positive_probability(model, X_val)
    return select_threshold(y_val, probabilities, recall_target)


def evaluate_profile(X_train, y_train, X_test, y_test, raw_features, profile, recall_target):
    threshold = inner_threshold(X_train, y_train, raw_features, profile, recall_target)
    model, calibrated = build_calibrated_model(raw_features, profile, y_train)
    model.fit(X_train, y_train)
    probabilities = predict_positive_probability(model, X_test)
    metrics = classification_metrics(y_test, probabilities, threshold)
    metrics["calibrated"] = calibrated
    return model, metrics, probabilities


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="BASE_DE_DADOS_PEDE_2024_-_DATATHON.xlsx")
    parser.add_argument("--out", default="model_risco_defasagem.pkl")
    parser.add_argument("--metadata-out", default="model_risco_defasagem.metadata.json")
    parser.add_argument("--target-mode", choices=TARGET_MODES, default="future_defasagem")
    parser.add_argument("--model-profile", choices=["complete", "actionable"], default="complete")
    parser.add_argument("--recall-target", type=float, default=0.80)
    args = parser.parse_args()

    dataset, _, raw_features, diagnostics = build_dataset(args.data, args.target_mode)
    train_period = int(dataset["Ano_N"].min())
    test_period = int(dataset["Ano_N"].max())
    train = dataset[dataset["Ano_N"] == train_period].copy()
    test = dataset[dataset["Ano_N"] == test_period].copy()
    if train.empty or test.empty:
        raise ValueError("Não foi possível formar o holdout temporal.")
    if train["risco_defasagem"].nunique() < 2 or test["risco_defasagem"].nunique() < 2:
        raise ValueError("Treino e teste temporal precisam conter as duas classes.")

    X_train, y_train = train[raw_features], train["risco_defasagem"]
    X_test, y_test = test[raw_features], test["risco_defasagem"]

    comparison = {}
    evaluated_models = {}
    for profile in ["complete", "actionable"]:
        model, metrics, probabilities = evaluate_profile(
            X_train, y_train, X_test, y_test, raw_features, profile, args.recall_target
        )
        comparison[profile] = metrics
        evaluated_models[profile] = (model, probabilities)

    # Baselines avaliados no mesmo período temporal.
    prior_probabilities = np.full(len(y_test), float(y_train.mean()))
    comparison["baseline_prior"] = classification_metrics(y_test, prior_probabilities, 0.50)
    persistence_probabilities = X_test["Defasagem_N"].lt(0).astype(float).to_numpy()
    comparison["baseline_persistence"] = classification_metrics(y_test, persistence_probabilities, 0.50)

    selected_eval_model, selected_test_probabilities = evaluated_models[args.model_profile]
    selected_metrics = comparison[args.model_profile]
    threshold = float(selected_metrics["threshold"])
    risk_bands = {
        "low_max": round(max(0.05, threshold / 2), 4),
        "priority_min": round(threshold, 4),
    }

    importance = permutation_importance(
        selected_eval_model,
        X_test,
        y_test,
        scoring="average_precision",
        n_repeats=15,
        random_state=42,
        n_jobs=-1,
    )
    global_importance = {
        feature: round(float(value), 5)
        for feature, value in sorted(
            zip(raw_features, importance.importances_mean), key=lambda item: item[1], reverse=True
        )
    }

    # O modelo de produção usa todos os pares somente após a avaliação temporal.
    X_all, y_all = dataset[raw_features], dataset["risco_defasagem"]
    final_model, calibrated = build_calibrated_model(raw_features, args.model_profile, y_all)
    final_model.fit(X_all, y_all)

    input_ranges = {
        feature: {
            "observed_min": None if X_all[feature].dropna().empty else float(X_all[feature].min()),
            "observed_max": None if X_all[feature].dropna().empty else float(X_all[feature].max()),
            "observed_mean": None if X_all[feature].dropna().empty else float(X_all[feature].mean()),
            "hard_min": HARD_LIMITS[feature][0],
            "hard_max": HARD_LIMITS[feature][1],
        }
        for feature in raw_features
    }

    artifact = {
        "artifact_version": 2,
        "model": final_model,
        "raw_features": raw_features,
        "model_profile": args.model_profile,
        "calibrated": calibrated,
        "target_mode": args.target_mode,
        "target_definition": TARGET_MODES[args.target_mode],
        "threshold": threshold,
        "risk_bands": risk_bands,
        "metrics": {
            "temporal_holdout": selected_metrics,
            "model_comparison": comparison,
            "train_period": f"{train_period}->{train_period + 1}",
            "test_period": f"{test_period}->{test_period + 1}",
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "n_modeling": int(len(dataset)),
        },
        "global_permutation_importance": global_importance,
        "input_ranges": input_ranges,
        "dataset_diagnostics": diagnostics,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "versions": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
    }

    joblib.dump(artifact, args.out)
    metadata = {key: value for key, value in artifact.items() if key != "model"}
    Path(args.metadata_out).write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(f"\nModelo salvo em: {args.out}")


if __name__ == "__main__":
    main()
