import numpy as np
import pandas as pd

from model_utils import (
    FeatureBuilder,
    RiskBands,
    classify_risk,
    to_numeric_series,
    validate_prediction_batch,
)


RAW = [
    "IAA_N",
    "IEG_N",
    "IPS_N",
    "IDA_N",
    "IAN_N",
    "IPV_N",
    "Defasagem_N",
    "Fase_num",
]


def sample_frame():
    return pd.DataFrame(
        {
            "IAA_N": ["7,5", 6],
            "IEG_N": [8, 5],
            "IPS_N": [7, np.nan],
            "IDA_N": [6, 4],
            "IAN_N": [10, 5],
            "IPV_N": [7, 5],
            "Defasagem_N": [0, -1],
            "Fase_num": [3, 4],
        }
    )


def test_numeric_conversion_accepts_comma_decimal():
    result = to_numeric_series(pd.Series(["7,5", " 2 ", "inválido"]))
    assert result.iloc[0] == 7.5
    assert result.iloc[1] == 2.0
    assert pd.isna(result.iloc[2])


def test_complete_profile_creates_features():
    builder = FeatureBuilder(RAW, profile="complete")
    result = builder.fit_transform(sample_frame())
    assert "media_indicadores" in result.columns
    assert "gap_iaa_ida" in result.columns
    assert "Defasagem_N" in result.columns


def test_actionable_profile_removes_direct_state_proxies():
    builder = FeatureBuilder(RAW, profile="actionable")
    result = builder.fit_transform(sample_frame())
    assert "IAN_N" not in result.columns
    assert "Defasagem_N" not in result.columns
    assert "gap_ipp_ian" not in result.columns


def test_risk_classification_uses_artifact_bands():
    bands = RiskBands(low_max=0.25, priority_min=0.50)
    assert classify_risk(0.10, bands) == "Baixo - rotina"
    assert classify_risk(0.30, bands) == "Monitoramento"
    assert classify_risk(0.70, bands) == "Prioritário"


def test_batch_validation_accepts_missing_cells_for_pipeline_imputation():
    frame = sample_frame().iloc[[1]].copy()
    ranges = {name: {"hard_min": 0, "hard_max": 10} for name in RAW}
    ranges["Defasagem_N"] = {"hard_min": -10, "hard_max": 10}
    result, valid = validate_prediction_batch(frame, RAW, ranges)
    assert valid.iloc[0]
    assert result["status_processamento"].iloc[0] == "válido com imputação/ajuste"
    assert "IPS_N será imputado" in result["avisos_processamento"].iloc[0]


def test_batch_validation_clips_only_small_boundary_residue():
    frame = sample_frame().iloc[[0]].copy()
    frame["IPV_N"] = 10.01
    ranges = {name: {"hard_min": 0, "hard_max": 10} for name in RAW}
    ranges["Defasagem_N"] = {"hard_min": -10, "hard_max": 10}
    result, valid = validate_prediction_batch(frame, RAW, ranges)
    assert valid.iloc[0]
    assert result["IPV_N"].iloc[0] == 10.0

    frame["IPV_N"] = 10.5
    result, valid = validate_prediction_batch(frame, RAW, ranges)
    assert not valid.iloc[0]
    assert "fora de [0, 10]" in result["motivo_erro"].iloc[0]
