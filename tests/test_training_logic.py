import pandas as pd

from train_model import longitudinal_pairs, select_threshold


def year_frame(year, defasagens):
    rows = []
    for index, defasagem in enumerate(defasagens, start=1):
        rows.append(
            {
                "RA": f"RA{index}",
                "Ano": year,
                "IAA": 7.0,
                "IEG": 7.0,
                "IPS": 7.0,
                "IPP": 7.0,
                "IDA": 7.0,
                "IAN": 10.0,
                "IPV": 7.0,
                "Defasagem": defasagem,
                "Fase_num": 3,
            }
        )
    return pd.DataFrame(rows)


def test_longitudinal_pairs_preserve_student_and_period():
    full = pd.concat([year_frame(2022, [0, -1]), year_frame(2023, [-1, 0])], ignore_index=True)
    result = longitudinal_pairs(full, 2022, 2023)
    assert list(result["RA"]) == ["RA1", "RA2"]
    assert set(result["Ano_N"]) == {2022}
    assert set(result["Ano_N1"]) == {2023}
    assert result.loc[result["RA"] == "RA1", "Defasagem_N1"].iloc[0] == -1


def test_threshold_is_selected_from_probabilities():
    y = pd.Series([0, 0, 1, 1])
    probabilities = [0.1, 0.2, 0.6, 0.9]
    threshold = select_threshold(y, probabilities, recall_target=1.0)
    assert 0.2 < threshold <= 0.6
