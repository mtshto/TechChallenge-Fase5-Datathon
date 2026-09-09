"""
Passos Mágicos — Painel Preditivo de Risco de Defasagem
Aplicação Streamlit que disponibiliza o modelo treinado (RandomForest, AUC ~0.87)
para uso da equipe pedagógica: predição individual e predição em lote.
"""
import io
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------- CONFIG
st.set_page_config(
    page_title="Passos Mágicos | Risco de Defasagem",
    page_icon="🔮",
    layout="wide",
)

MODEL_PATH = "model_risco_defasagem.pkl"

NOMES_AMIGAVEIS = {
    "IAA_N": "IAA — Autoavaliação",
    "IEG_N": "IEG — Engajamento",
    "IPS_N": "IPS — Psicossocial",
    "IDA_N": "IDA — Desempenho Acadêmico",
    "IAN_N": "IAN — Adequação de Nível",
    "IPV_N": "IPV — Ponto de Virada",
    "Defasagem_N": "Defasagem atual (anos)",
    "Fase_num": "Fase atual",
}


@st.cache_resource
def carregar_modelo(path=MODEL_PATH):
    return joblib.load(path)


def classificar_risco(p):
    if p < 0.33:
        return "🟢 Baixo", "#2e7d32"
    if p < 0.66:
        return "🟡 Médio", "#f9a825"
    return "🔴 Alto", "#c62828"


def gerar_template_csv(features):
    df = pd.DataFrame(
        {
            "RA": ["RA-exemplo-1", "RA-exemplo-2"],
            "IAA_N": [8.0, 5.5],
            "IEG_N": [7.5, 4.0],
            "IPS_N": [6.8, 5.0],
            "IDA_N": [7.0, 4.5],
            "IAN_N": [10.0, 5.0],
            "IPV_N": [7.2, 4.8],
            "Defasagem_N": [0, -1],
            "Fase_num": [3, 4],
        }
    )
    return df.to_csv(index=False).encode("utf-8")


artifact = carregar_modelo()
model = artifact["model"]
FEATURES = artifact["features"]
metrics = artifact["metrics"]
importancias = artifact["feature_importances"]
ranges = artifact["feature_ranges"]

# --------------------------------------------------------------------------- SIDEBAR
st.sidebar.title("🔮 Passos Mágicos")
st.sidebar.caption("Painel preditivo de risco de defasagem")
pagina = st.sidebar.radio(
    "Navegação",
    ["Predição individual", "Predição em lote (upload)", "Sobre o modelo"],
)
st.sidebar.markdown("---")
st.sidebar.metric("AUC do modelo (validação cruzada)", f"{metrics['auc_cv_mean']:.2f}")
st.sidebar.metric("Acurácia (holdout)", f"{metrics['accuracy_holdout']:.0%}")
st.sidebar.caption(f"Treinado em: {artifact['trained_at'][:10]}")

# --------------------------------------------------------------------------- PÁGINA 1
if pagina == "Predição individual":
    st.title("Predição individual de risco de defasagem")
    st.markdown(
        """Informe os indicadores **atuais** do aluno (ano N). O modelo estima a
        probabilidade de o aluno apresentar **defasagem de nível (Fase efetiva < Fase ideal)
        no ano seguinte**, com base em padrões observados nas turmas de 2022–2024."""
    )

    with st.form("form_individual"):
        c1, c2 = st.columns(2)
        with c1:
            iaa = st.slider("IAA — Autoavaliação", 0.0, 10.0, 7.0, 0.1)
            ieg = st.slider("IEG — Engajamento", 0.0, 10.0, 7.0, 0.1)
            ips = st.slider("IPS — Psicossocial", 0.0, 10.0, 7.0, 0.1)
            ida = st.slider("IDA — Desempenho Acadêmico", 0.0, 10.0, 6.5, 0.1)
        with c2:
            ian = st.slider("IAN — Adequação de Nível", 0.0, 10.0, 7.5, 0.5)
            ipv = st.slider("IPV — Ponto de Virada", 0.0, 10.0, 6.5, 0.1)
            defasagem = st.number_input("Defasagem atual (anos; negativo = atrasado)", -5, 3, 0, 1)
            fase = st.number_input("Fase atual (0 = ALFA)", 0, 9, 3, 1)

        enviado = st.form_submit_button("Calcular risco", type="primary")

    if enviado:
        entrada = pd.DataFrame([[iaa, ieg, ips, ida, ian, ipv, defasagem, fase]], columns=FEATURES)
        proba = model.predict_proba(entrada)[0, 1]
        label, cor = classificar_risco(proba)

        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("Probabilidade de defasagem no próximo ano", f"{proba:.0%}")
            st.markdown(f"### Classificação: <span style='color:{cor}'>{label}</span>", unsafe_allow_html=True)
            st.progress(min(max(proba, 0.0), 1.0))

        with col2:
            if proba >= 0.66:
                st.error(
                    "**Ação recomendada:** priorizar para acompanhamento pedagógico e "
                    "psicossocial imediato. Reforço acadêmico direcionado e checagem de IPS/IEG."
                )
            elif proba >= 0.33:
                st.warning(
                    "**Ação recomendada:** monitorar de perto na próxima avaliação. "
                    "Reforçar engajamento (IEG) e checar consistência entre IDA e IAN."
                )
            else:
                st.success(
                    "**Ação recomendada:** manter acompanhamento de rotina — indicadores "
                    "atuais sugerem baixo risco de defasagem no próximo ciclo."
                )

        st.markdown("#### Contribuição de cada indicador (importância global do modelo)")
        imp_series = pd.Series(importancias).sort_values()
        imp_series.index = [NOMES_AMIGAVEIS[i] for i in imp_series.index]
        fig, ax = plt.subplots(figsize=(7, 3.5))
        imp_series.plot(kind="barh", ax=ax, color="#6a4c93")
        ax.set_xlabel("Importância no modelo")
        st.pyplot(fig)

# --------------------------------------------------------------------------- PÁGINA 2
elif pagina == "Predição em lote (upload)":
    st.title("Predição em lote")
    st.markdown(
        """Envie uma planilha (CSV ou XLSX) com um aluno por linha e as colunas de
        indicadores listadas abaixo. Baixe o modelo de planilha se precisar de referência."""
    )

    st.download_button(
        "⬇️ Baixar modelo de planilha (CSV)",
        data=gerar_template_csv(FEATURES),
        file_name="template_predicao_passos_magicos.csv",
        mime="text/csv",
    )

    st.caption(
        "Colunas esperadas: `RA` (opcional), " + ", ".join(f"`{f}`" for f in FEATURES)
    )

    arquivo = st.file_uploader("Arquivo de alunos", type=["csv", "xlsx"])

    if arquivo is not None:
        try:
            if arquivo.name.endswith(".csv"):
                df_in = pd.read_csv(arquivo)
            else:
                df_in = pd.read_excel(arquivo)
        except Exception as e:
            st.error(f"Não foi possível ler o arquivo: {e}")
            st.stop()

        faltando = [f for f in FEATURES if f not in df_in.columns]
        if faltando:
            st.error(
                "As seguintes colunas obrigatórias não foram encontradas no arquivo: "
                + ", ".join(faltando)
                + ". Use o modelo de planilha acima como referência."
            )
            st.stop()

        df_score = df_in.copy()
        validos = df_score[FEATURES].notna().all(axis=1)
        n_invalidos = (~validos).sum()
        if n_invalidos:
            st.warning(f"{n_invalidos} linha(s) com valores ausentes serão ignoradas na predição.")

        df_valid = df_score[validos].copy()
        proba = model.predict_proba(df_valid[FEATURES])[:, 1]
        df_valid["probabilidade_risco"] = proba.round(3)
        df_valid["classificacao_risco"] = [classificar_risco(p)[0] for p in proba]
        df_valid = df_valid.sort_values("probabilidade_risco", ascending=False)

        st.success(f"{len(df_valid)} aluno(s) avaliado(s).")

        c1, c2, c3 = st.columns(3)
        c1.metric("Risco alto (≥66%)", int((proba >= 0.66).sum()))
        c2.metric("Risco médio (33–66%)", int(((proba >= 0.33) & (proba < 0.66)).sum()))
        c3.metric("Risco baixo (<33%)", int((proba < 0.33).sum()))

        cols_mostrar = (["RA"] if "RA" in df_valid.columns else []) + FEATURES + [
            "probabilidade_risco",
            "classificacao_risco",
        ]
        st.dataframe(df_valid[cols_mostrar], use_container_width=True, height=420)

        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.hist(proba, bins=20, color="#6a4c93", edgecolor="white")
        ax.axvline(0.33, color="orange", linestyle="--", label="limite médio")
        ax.axvline(0.66, color="red", linestyle="--", label="limite alto")
        ax.set_xlabel("Probabilidade de risco")
        ax.set_ylabel("Nº de alunos")
        ax.legend()
        st.pyplot(fig)

        csv_saida = df_valid[cols_mostrar].to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Baixar resultado completo (CSV)",
            data=csv_saida,
            file_name=f"predicao_risco_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
        )

# --------------------------------------------------------------------------- PÁGINA 3
else:
    st.title("Sobre o modelo")
    st.markdown(
        """
Este modelo (**Random Forest**) foi treinado com dados da Pesquisa Extensiva do
Desenvolvimento Educacional (**PEDE**) de 2022, 2023 e 2024 da Associação Passos Mágicos.

**Definição do alvo:** o modelo estima a probabilidade de o aluno apresentar
**defasagem de nível (Fase efetiva abaixo da Fase ideal) no ano seguinte**, a partir
dos indicadores do ano corrente — permitindo agir *antes* de a defasagem se concretizar.
"""
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("AUC (validação cruzada, 5-fold)", f"{metrics['auc_cv_mean']:.3f} ± {metrics['auc_cv_std']:.3f}")
    c2.metric("AUC (conjunto de teste)", f"{metrics['auc_holdout']:.3f}")
    c3.metric("Precisão — classe 'em risco'", f"{metrics['precision_em_risco']:.0%}")
    c4.metric("Recall — classe 'em risco'", f"{metrics['recall_em_risco']:.0%}")

    st.markdown(f"""
- **Amostra de treino:** {metrics['n_treino']} pares aluno-ano (RA presente em dois anos consecutivos).
- **Taxa-base de risco na amostra:** {metrics['taxa_risco_base']:.0%}.
- **Tipo de modelo:** {artifact['model_type']}.
- **Data do treinamento:** {artifact['trained_at']}.
""")

    st.markdown("#### Importância das variáveis")
    imp_series = pd.Series(importancias).sort_values()
    imp_series.index = [NOMES_AMIGAVEIS[i] for i in imp_series.index]
    fig, ax = plt.subplots(figsize=(8, 4))
    imp_series.plot(kind="barh", ax=ax, color="#6a4c93")
    ax.set_xlabel("Importância no modelo")
    st.pyplot(fig)

    st.markdown("#### Faixa de valores observada no treinamento (referência)")
    df_ranges = pd.DataFrame(ranges).T.round(2)
    df_ranges.index = [NOMES_AMIGAVEIS[i] for i in df_ranges.index]
    st.dataframe(df_ranges, use_container_width=True)

    st.info(
        "**Limitações:** o modelo foi treinado com alunos que permaneceram no programa "
        "em dois anos consecutivos (~1.290 pares); alunos novos ou com dados incompletos "
        "não fazem parte do treinamento. Recomenda-se retreinar o modelo periodicamente "
        "(script `train_model.py`) à medida que novos ciclos do PEDE forem concluídos."
    )
