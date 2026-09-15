"""Componentes compartilhados entre treinamento e aplicação Streamlit.

Este módulo precisa acompanhar o arquivo ``model_risco_defasagem.pkl`` no deploy,
pois o Pipeline serializado utiliza ``FeatureBuilder``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


BASE_RAW_FEATURES = [
    "IAA_N",
    "IEG_N",
    "IPS_N",
    "IDA_N",
    "IAN_N",
    "IPV_N",
    "Defasagem_N",
    "Fase_num",
]

OPTIONAL_RAW_FEATURES = ["IPP_N"]

FEATURE_LABELS = {
    "IAA_N": "IAA - Autoavaliação",
    "IEG_N": "IEG - Engajamento",
    "IPS_N": "IPS - Psicossocial",
    "IPP_N": "IPP - Psicopedagógico",
    "IDA_N": "IDA - Desempenho Acadêmico",
    "IAN_N": "IAN - Adequação de Nível",
    "IPV_N": "IPV - Ponto de Virada",
    "Defasagem_N": "Defasagem atual (anos)",
    "Fase_num": "Fase atual",
}

HARD_LIMITS = {
    "IAA_N": (0.0, 10.0),
    "IEG_N": (0.0, 10.0),
    "IPS_N": (0.0, 10.0),
    "IPP_N": (0.0, 10.0),
    "IDA_N": (0.0, 10.0),
    "IAN_N": (0.0, 10.0),
    "IPV_N": (0.0, 10.0),
    "Defasagem_N": (-10.0, 10.0),
    "Fase_num": (0.0, 10.0),
}

BOUNDARY_TOLERANCE = 0.02


def to_numeric_series(series: pd.Series) -> pd.Series:
    """Converte números, inclusive textos com vírgula decimal, para float."""

    cleaned = (
        series.astype("string")
        .str.strip()
        .str.replace("\u00a0", "", regex=False)
        .str.replace(",", ".", regex=False)
    )
    return pd.to_numeric(cleaned, errors="coerce")


def prepare_numeric_frame(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Retorna cópia com as colunas solicitadas convertidas para numérico."""

    out = df.copy()
    for column in columns:
        if column not in out.columns:
            out[column] = np.nan
        out[column] = to_numeric_series(out[column])
    return out


def validate_prediction_batch(
    df: pd.DataFrame,
    features: list[str],
    ranges: dict,
    boundary_tolerance: float = BOUNDARY_TOLERANCE,
) -> tuple[pd.DataFrame, pd.Series]:
    """Valida entradas em lote sem descartar linhas silenciosamente.

    Células ausentes são aceitas porque o Pipeline contém imputação treinada.
    Colunas inteiras ausentes, textos não numéricos, valores realmente fora da
    escala e RAs duplicados impedem a predição. Pequenos resíduos de ponto
    flutuante junto aos limites (por exemplo, 10,01 em uma escala 0--10) são
    ajustados para o limite e registrados como aviso.
    """

    output = df.copy()
    errors = pd.Series("", index=output.index, dtype="string")
    warnings = pd.Series("", index=output.index, dtype="string")

    def append_message(target: pd.Series, mask: pd.Series, message: str) -> None:
        empty = target.eq("")
        target.loc[mask & empty] = message
        target.loc[mask & ~empty] = target.loc[mask & ~empty] + "; " + message

    for feature in features:
        if feature not in output.columns:
            output[feature] = np.nan
            append_message(errors, pd.Series(True, index=output.index), f"coluna ausente: {feature}")
            continue

        original = output[feature]
        converted = to_numeric_series(original)
        invalid_numeric = original.notna() & converted.isna()
        missing = converted.isna() & ~invalid_numeric
        low = float(ranges[feature]["hard_min"])
        high = float(ranges[feature]["hard_max"])

        near_low = converted.notna() & converted.lt(low) & converted.ge(low - boundary_tolerance)
        near_high = converted.notna() & converted.gt(high) & converted.le(high + boundary_tolerance)
        outside = converted.notna() & (
            converted.lt(low - boundary_tolerance) | converted.gt(high + boundary_tolerance)
        )

        append_message(errors, invalid_numeric, f"{feature} não numérico")
        append_message(errors, outside, f"{feature} fora de [{low:g}, {high:g}]")
        append_message(warnings, missing, f"{feature} será imputado")
        append_message(warnings, near_low | near_high, f"{feature} ajustado ao limite da escala")

        converted.loc[near_low] = low
        converted.loc[near_high] = high
        output[feature] = converted

    if "RA" in output.columns:
        duplicated = output["RA"].notna() & output["RA"].duplicated(keep=False)
        append_message(errors, duplicated, "RA duplicado no arquivo")

    valid = errors.eq("")
    status = np.select(
        [~valid, warnings.ne("")],
        ["inválido", "válido com imputação/ajuste"],
        default="válido",
    )
    output["status_processamento"] = status
    output["motivo_erro"] = errors
    output["avisos_processamento"] = warnings
    return output, valid


class FeatureBuilder(BaseEstimator, TransformerMixin):
    """Cria features derivadas dentro do Pipeline, evitando leakage de processo.

    ``profile='complete'`` usa a condição atual de adequação/defasagem.
    ``profile='actionable'`` remove IAN e Defasagem, permitindo medir quanto os
    demais indicadores antecipam risco sem depender diretamente do estado atual.
    """

    def __init__(self, raw_features: list[str], profile: str = "complete", critical_cutoff: float = 6.0):
        self.raw_features = raw_features
        self.profile = profile
        self.critical_cutoff = critical_cutoff

    def fit(self, X, y=None):
        if self.profile not in {"complete", "actionable"}:
            raise ValueError("profile deve ser 'complete' ou 'actionable'.")
        self.feature_names_out_ = list(self._transform(self._as_frame(X)).columns)
        return self

    def transform(self, X):
        return self._transform(self._as_frame(X))

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.feature_names_out_, dtype=object)

    def _as_frame(self, X) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            frame = X.copy()
        else:
            frame = pd.DataFrame(X, columns=self.raw_features)
        return prepare_numeric_frame(frame, self.raw_features)

    def _transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        excluded = {"IAN_N", "Defasagem_N"} if self.profile == "actionable" else set()
        selected = [c for c in self.raw_features if c not in excluded]
        out = frame[selected].copy()

        indicators = [
            c
            for c in ["IAA_N", "IEG_N", "IPS_N", "IPP_N", "IDA_N", "IAN_N", "IPV_N"]
            if c in out.columns
        ]
        if indicators:
            out["media_indicadores"] = out[indicators].mean(axis=1)
            out["desvio_indicadores"] = out[indicators].std(axis=1, ddof=0)
            out["min_indicadores"] = out[indicators].min(axis=1)
            out["max_indicadores"] = out[indicators].max(axis=1)
            out["n_indicadores_criticos"] = (out[indicators] < self.critical_cutoff).sum(axis=1)

        if {"IAA_N", "IDA_N"}.issubset(out.columns):
            out["gap_iaa_ida"] = out["IAA_N"] - out["IDA_N"]
        if {"IEG_N", "IDA_N"}.issubset(out.columns):
            out["gap_ieg_ida"] = out["IEG_N"] - out["IDA_N"]
        if {"IPS_N", "IDA_N"}.issubset(out.columns):
            out["gap_ips_ida"] = out["IPS_N"] - out["IDA_N"]
        if {"IPP_N", "IAN_N"}.issubset(out.columns):
            out["gap_ipp_ian"] = out["IPP_N"] - out["IAN_N"]

        return out.replace([np.inf, -np.inf], np.nan)


@dataclass(frozen=True)
class RiskBands:
    low_max: float
    priority_min: float

    @classmethod
    def from_mapping(cls, value: dict) -> "RiskBands":
        return cls(
            low_max=float(value["low_max"]),
            priority_min=float(value["priority_min"]),
        )


def classify_risk(probability: float, bands: RiskBands) -> str:
    if probability < bands.low_max:
        return "Baixo - rotina"
    if probability < bands.priority_min:
        return "Monitoramento"
    return "Prioritário"
