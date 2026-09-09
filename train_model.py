"""
Treina o modelo de risco de defasagem para a Associação Passos Mágicos
e salva o artefato (.pkl) usado pela aplicação Streamlit (app.py).

Uso:
    python train_model.py --data BASE_DE_DADOS_PEDE_2024_-_DATATHON.xlsx --out model_risco_defasagem.pkl
"""
import argparse
import re
import json
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, classification_report

FEATURES = ['IAA_N', 'IEG_N', 'IPS_N', 'IDA_N', 'IAN_N', 'IPV_N', 'Defasagem_N', 'Fase_num']


def fase_num(x):
    """Extrai o número da fase a partir de rótulos como 'FASE 3', '3A' ou 'ALFA' (=0)."""
    if pd.isna(x):
        return np.nan
    s = str(x).upper().strip()
    if s.startswith('ALFA'):
        return 0
    m = re.search(r'(\d+)', s)
    return int(m.group(1)) if m else np.nan


def load_year(path, sheet, ano):
    df = pd.read_excel(path, sheet_name=sheet)
    out = pd.DataFrame()
    out['RA'] = df['RA']
    out['Ano'] = ano
    out['Fase_num'] = df['Fase'].apply(fase_num)

    inde_col = {2022: 'INDE 22', 2023: 'INDE 2023', 2024: 'INDE 2024'}[ano]
    out['INDE'] = pd.to_numeric(df[inde_col], errors='coerce')

    pedra_col = 'Pedra 22' if ano == 2022 else f'Pedra {ano}'
    out['Pedra'] = df[pedra_col].replace({'Agata': 'Ágata'})
    out.loc[out['Pedra'] == 'INCLUIR', 'Pedra'] = np.nan

    out['IAA'] = df['IAA']
    out['IEG'] = df['IEG']
    out['IPS'] = df['IPS']
    out['IPP'] = df['IPP'] if 'IPP' in df.columns else np.nan
    out['IDA'] = df['IDA']
    out['IAN'] = df['IAN']
    out['IPV'] = df['IPV']
    out['Defasagem'] = df['Defas'] if 'Defas' in df.columns else df['Defasagem']
    return out


def pares_longitudinais(full, ano_n, ano_n1, cols):
    d1 = full[full['Ano'] == ano_n].set_index('RA')
    d2 = full[full['Ano'] == ano_n1].set_index('RA')
    comuns = d1.index.intersection(d2.index)
    out = pd.DataFrame(index=comuns)
    for c in cols:
        out[f'{c}_N'] = d1.loc[comuns, c]
        out[f'{c}_N1'] = d2.loc[comuns, c]
    return out


def build_dataset(path):
    d22 = load_year(path, 'PEDE2022', 2022)
    d23 = load_year(path, 'PEDE2023', 2023)
    d24 = load_year(path, 'PEDE2024', 2024)
    full = pd.concat([d22, d23, d24], ignore_index=True)

    cols_feat = ['IAA', 'IEG', 'IPS', 'IDA', 'IAN', 'IPV', 'Defasagem', 'Fase_num']
    pares = pd.concat([
        pares_longitudinais(full, 2022, 2023, cols_feat),
        pares_longitudinais(full, 2023, 2024, cols_feat),
    ])
    pares = pares.rename(columns={'Fase_num_N': 'Fase_num'})
    pares['risco_defasagem'] = (pares['Defasagem_N1'] <= -1).astype(int)

    dataset = pares[FEATURES + ['risco_defasagem']].dropna()
    return dataset, full


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='BASE_DE_DADOS_PEDE_2024_-_DATATHON.xlsx')
    ap.add_argument('--out', default='model_risco_defasagem.pkl')
    args = ap.parse_args()

    dataset, full = build_dataset(args.data)
    X, y = dataset[FEATURES], dataset['risco_defasagem']

    # --- avaliação (holdout) para reportar métricas honestas no app
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    eval_model = RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42)
    eval_model.fit(X_train, y_train)
    proba_test = eval_model.predict_proba(X_test)[:, 1]
    auc_holdout = roc_auc_score(y_test, proba_test)
    report = classification_report(y_test, eval_model.predict(X_test), output_dict=True)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(
        RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42),
        X, y, cv=cv, scoring='roc_auc'
    )

    # --- modelo final treinado com todos os dados disponíveis (produção)
    final_model = RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42)
    final_model.fit(X, y)

    importances = dict(zip(FEATURES, final_model.feature_importances_.round(4)))

    artifact = {
        'model': final_model,
        'features': FEATURES,
        'feature_importances': importances,
        'metrics': {
            'auc_holdout': round(float(auc_holdout), 4),
            'auc_cv_mean': round(float(cv_scores.mean()), 4),
            'auc_cv_std': round(float(cv_scores.std()), 4),
            'accuracy_holdout': round(float(report['accuracy']), 4),
            'precision_em_risco': round(float(report['1']['precision']), 4),
            'recall_em_risco': round(float(report['1']['recall']), 4),
            'n_treino': int(len(X)),
            'taxa_risco_base': round(float(y.mean()), 4),
        },
        'feature_ranges': {
            f: {'min': float(X[f].min()), 'max': float(X[f].max()), 'mean': float(X[f].mean())}
            for f in FEATURES
        },
        'trained_at': datetime.now().isoformat(timespec='seconds'),
        'target_definition': 'risco = 1 se Defasagem no ano seguinte (N+1) <= -1, usando indicadores do ano N',
        'model_type': 'RandomForestClassifier(n_estimators=300, max_depth=6)',
    }

    joblib.dump(artifact, args.out)
    print(json.dumps(artifact['metrics'], indent=2, ensure_ascii=False))
    print(f"\nModelo salvo em: {args.out}")


if __name__ == '__main__':
    main()
