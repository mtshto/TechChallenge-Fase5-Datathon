"""Aplicação Streamlit para apoio preventivo à equipe da Passos Mágicos."""

from __future__ import annotations

from datetime import datetime

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from model_utils import (
    FEATURE_LABELS,
    RiskBands,
    classify_risk,
    distribution_warnings,
    predict_positive_probability,
    prediction_field_guide,
    validate_prediction_batch,
)


st.set_page_config(
    page_title="Passos Mágicos | Risco de Defasagem",
    page_icon="🔮",
    layout="wide",
)

LOGO_PATH = Path(__file__).parent / "assets" / "passos_magicos.png"

st.image(LOGO_PATH, width=280)


MODEL_PATH = "model_risco_defasagem.pkl"


@st.cache_resource(show_spinner="Carregando modelo...")
def load_artifact(path=MODEL_PATH):
    artifact = joblib.load(path)
    if artifact.get("artifact_version", 0) < 2:
        raise ValueError(
            "O modelo foi gerado pela versão antiga do treinamento. "
            "Execute novamente: python train_model.py --data <arquivo.xlsx>."
        )
    required = {
        "model",
        "raw_features",
        "target_definition",
        "threshold",
        "risk_bands",
        "metrics",
        "input_ranges",
    }
    missing = required - set(artifact)
    if missing:
        raise ValueError(f"Artefato incompleto. Chaves ausentes: {sorted(missing)}")
    return artifact


def read_uploaded_file(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".xlsx"):
        return pd.read_excel(uploaded_file)
    try:
        return pd.read_csv(uploaded_file, sep=None, engine="python")
    except Exception:
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file, sep=";")


def input_widget(feature: str, ranges: dict):
    label = FEATURE_LABELS.get(feature, feature)
    limits = ranges[feature]
    lower = float(limits["hard_min"])
    upper = float(limits["hard_max"])
    observed_mean = limits.get("observed_mean")
    default = float(observed_mean) if observed_mean is not None else (lower + upper) / 2
    default = min(max(default, lower), upper)
    observed_min = limits.get("observed_min")
    observed_max = limits.get("observed_max")
    help_text = None
    if observed_min is not None and observed_max is not None:
        help_text = (
            f"Faixa observada no treinamento: {float(observed_min):g} a "
            f"{float(observed_max):g}. Valores além dela geram alerta de extrapolação."
        )

    if feature in {"Defasagem_N", "Fase_num"}:
        return st.number_input(
            label,
            int(lower),
            int(upper),
            int(round(default)),
            1,
            help=help_text,
        )
    step = 0.5 if feature == "IAN_N" else 0.1
    return st.slider(label, lower, upper, default, step, help=help_text)


def personalized_recommendations(row: pd.Series) -> list[str]:
    recommendations = []
    if row.get("IDA_N", 10) < 6 or row.get("IAN_N", 10) < 6:
        recommendations.append("Revisar necessidades pedagógicas e considerar reforço direcionado.")
    if row.get("IEG_N", 10) < 6:
        recommendations.append("Investigar participação, frequência e vínculo com as atividades.")
    if row.get("IPS_N", 10) < 6:
        recommendations.append("Sugerir avaliação psicossocial pela equipe responsável.")
    if "IPP_N" in row and pd.notna(row["IPP_N"]) and row["IPP_N"] < 6:
        recommendations.append("Revisar a avaliação psicopedagógica e seu plano de acompanhamento.")
    if pd.notna(row.get("IAA_N")) and pd.notna(row.get("IDA_N")) and row["IAA_N"] - row["IDA_N"] > 2:
        recommendations.append("Conversar sobre a diferença entre autoavaliação e desempenho observado.")
    return recommendations or ["Manter acompanhamento de rotina e atualizar os indicadores no próximo ciclo."]


def build_template(features: list[str], ranges: dict) -> bytes:
    row = {"RA": "RA-exemplo-1"}
    for feature in features:
        mean = ranges[feature].get("observed_mean")
        row[feature] = round(float(mean), 2) if mean is not None else 0
    return pd.DataFrame([row]).to_csv(index=False, sep=";").encode("utf-8-sig")


def show_global_importance(artifact):
    values = artifact.get("global_permutation_importance", {})

    if not values:
        st.info("Importância global não disponível no artefato.")
        return

    series = pd.Series(values).sort_values(ascending=True)

    # Usa os nomes amigáveis definidos no model_utils.py
    series.index = [
        FEATURE_LABELS.get(name, name)
        for name in series.index
    ]

    theme = st.context.theme.type

    if theme == "dark":
        axis_color = "white"
    else:
        axis_color = "black"

    fig, ax = plt.subplots(figsize=(10, 5))

    # Fundo transparente
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")

    # Gráfico horizontal
    series.plot(
        kind="barh",
        ax=ax,
        color="#4C78A8",
        width=0.7,
    )

    # Labels
    ax.set_xlabel(
        "Queda média de PR-AUC",
        color=axis_color,
    )

    ax.set_ylabel(
        "Variável",
        color=axis_color,
    )

    # Eixo X
    ax.tick_params(
        axis="x",
        colors=axis_color,
    )

    # Eixo Y
    ax.tick_params(
        axis="y",
        colors=axis_color,
    )

    # Bordas
    for spine in ax.spines.values():
        spine.set_color(axis_color)

    # Grade apenas no eixo X
    ax.grid(
        axis="x",
        alpha=0.2,
    )

    plt.tight_layout()

    st.pyplot(fig, use_container_width=True)

    plt.close(fig)

    st.caption(
        "Importância global por permutação no teste temporal. "
        "Este gráfico não explica a decisão de um aluno específico."
    )

try:
    artifact = load_artifact()
except FileNotFoundError:
    st.error(f"Arquivo `{MODEL_PATH}` não encontrado. Treine o modelo antes do deploy.")
    st.stop()
except Exception as error:
    st.error(f"Não foi possível carregar o modelo: {type(error).__name__}: {error}")
    st.stop()

model = artifact["model"]
FEATURES = artifact["raw_features"]
RANGES = artifact["input_ranges"]
BANDS = RiskBands.from_mapping(artifact["risk_bands"])
METRICS = artifact["metrics"]["temporal_holdout"]

st.sidebar.title("🔮 Passos Mágicos")
st.sidebar.caption("Apoio preventivo - decisão final sempre humana")
page = st.sidebar.radio("Navegação", ["Predição individual", "Predição em lote", "Sobre o modelo"])
st.sidebar.markdown("---")
st.sidebar.metric("PR-AUC temporal", f"{METRICS['pr_auc']:.3f}")
st.sidebar.metric("Recall no threshold", f"{METRICS['recall']:.0%}")
st.sidebar.caption(f"Threshold prioritário: {artifact['threshold']:.3f}")

if page == "Predição individual":
    st.title("Predição individual de risco de defasagem")
    st.write(artifact["target_definition"])
    st.info(
        "Nos indicadores, **zero é uma nota real**, não um campo ausente. "
        "Se a avaliação não existir, utilize a predição em lote e deixe a célula vazia "
        "para que o Pipeline aplique a imputação aprendida no treinamento."
    )
    with st.form("individual"):
        values = {}
        columns = st.columns(2)
        for index, feature in enumerate(FEATURES):
            with columns[index % 2]:
                values[feature] = input_widget(feature, RANGES)
        submitted = st.form_submit_button("Calcular risco", type="primary")

    if submitted:
        frame = pd.DataFrame([values], columns=FEATURES)
        probability = float(predict_positive_probability(model, frame)[0])
        classification = classify_risk(probability, BANDS)
        distribution_message = distribution_warnings(frame, FEATURES, RANGES).iloc[0]
        if distribution_message:
            st.warning(
                "Esta combinação contém valores fora da faixa observada no treinamento. "
                "A probabilidade foi calculada, mas deve ser tratada como uma extrapolação "
                f"de menor confiabilidade. Detalhes: {distribution_message}."
            )
        c1, c2 = st.columns([1, 2])
        c1.metric("Probabilidade calibrada", f"{probability:.1%}")
        c1.metric("Faixa operacional", classification)
        c1.progress(float(np.clip(probability, 0, 1)))
        threshold = float(artifact["threshold"])
        difference = probability - threshold
        if difference >= 0:
            c1.caption(
                f"Limite prioritário: {threshold:.1%}. A estimativa ficou "
                f"{difference:.1%} acima desse limite."
            )
        else:
            c1.caption(
                f"Limite prioritário: {threshold:.1%}. A estimativa ficou "
                f"{abs(difference):.1%} abaixo desse limite."
            )
        c1.caption(
            "Probabilidade calibrada não é uma soma das notas: ela representa o risco "
            "estimado a partir de padrões históricos semelhantes."
        )
        with c2:
            st.subheader("Orientações para avaliação humana")
            for recommendation in personalized_recommendations(frame.iloc[0]):
                st.write(f"- {recommendation}")
        st.subheader("Importância global do modelo")
        show_global_importance(artifact)

elif page == "Predição em lote":
    st.title("Predição em lote")
    st.write("O resultado preserva linhas inválidas e informa o motivo do erro.")
    st.subheader("Campos necessários no arquivo")
    st.info(
        "Use exatamente os nomes de coluna apresentados abaixo. Todas as colunas dos "
        "indicadores são obrigatórias, mas células isoladas podem ficar vazias: nesses "
        "casos, o Pipeline aplicará a imputação aprendida no treinamento. A coluna RA é "
        "opcional, porém recomendada para identificar cada aluno."
    )
    st.dataframe(
        prediction_field_guide(FEATURES, RANGES),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "Exemplos: ALFA deve ser Fase_num = 0; Defasagem_N = -1 representa um ano "
        "de defasagem. Para evitar erros de digitação, baixe o template pronto."
    )
    st.download_button(
        "Baixar template CSV",
        build_template(FEATURES, RANGES),
        "template_predicao_passos_magicos.csv",
        "text/csv",
    )
    uploaded = st.file_uploader("Envie CSV ou XLSX", type=["csv", "xlsx"])
    if uploaded is not None:
        try:
            original = read_uploaded_file(uploaded)
        except Exception as error:
            st.error(f"Não foi possível ler o arquivo: {error}")
            st.stop()
        if original.empty:
            st.error("O arquivo não possui linhas para processamento.")
            st.stop()

        result, valid = validate_prediction_batch(original, FEATURES, RANGES)
        result["probabilidade_risco"] = np.nan
        result["classificacao_risco"] = pd.NA
        if valid.any():
            probabilities = predict_positive_probability(model, result.loc[valid, FEATURES])
            result.loc[valid, "probabilidade_risco"] = np.round(probabilities, 4)
            result.loc[valid, "classificacao_risco"] = [classify_risk(p, BANDS) for p in probabilities]
        else:
            st.warning("Nenhuma linha válida foi encontrada; consulte a coluna motivo_erro.")

        c1, c2, c3 = st.columns(3)
        c1.metric("Linhas recebidas", len(result))
        c2.metric("Linhas válidas", int(valid.sum()))
        c3.metric("Linhas inválidas", int((~valid).sum()))
        st.dataframe(result, use_container_width=True, height=460)

        export = result.to_csv(index=False, sep=";").encode("utf-8-sig")
        st.download_button(
            "Baixar resultado completo",
            export,
            f"predicao_risco_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            "text/csv",
        )

else:
    st.title("Sobre o modelo")
    st.markdown(f"**Target:** {artifact['target_definition']}")
    st.markdown(f"**Perfil:** `{artifact['model_profile']}`")
    st.markdown(f"**Calibrado:** {'sim' if artifact['calibrated'] else 'não'}")
    st.markdown(
        f"**Avaliação temporal:** treino {artifact['metrics']['train_period']} e "
        f"teste {artifact['metrics']['test_period']}"
    )

    metric_cols = st.columns(5)
    metric_cols[0].metric("ROC-AUC", f"{METRICS['roc_auc']:.3f}")
    metric_cols[1].metric("PR-AUC", f"{METRICS['pr_auc']:.3f}")
    metric_cols[2].metric("Recall", f"{METRICS['recall']:.0%}")
    metric_cols[3].metric("Precisão", f"{METRICS['precision']:.0%}")
    metric_cols[4].metric("F1", f"{METRICS['f1']:.3f}")

    st.subheader("Comparação de modelos")
    comparison = pd.DataFrame(artifact["metrics"]["model_comparison"]).T
    columns = [c for c in ["roc_auc", "pr_auc", "recall", "precision", "f1", "brier"] if c in comparison]
    st.dataframe(comparison[columns], use_container_width=True)

    st.subheader("Importância global")
    show_global_importance(artifact)

    st.warning(
        "Este modelo é uma ferramenta de apoio. A pontuação não deve produzir decisões "
        "automáticas, punições ou exclusão de oportunidades. Os casos sinalizados precisam "
        "ser avaliados pela equipe pedagógica e psicossocial."
    )
