# Datathon Passos Mágicos - Risco de Defasagem

Projeto de Data Science desenvolvido para apoiar a Associação Passos Mágicos na identificação antecipada de alunos que podem apresentar defasagem de nível no ciclo seguinte.

> O modelo é uma ferramenta de apoio. Nenhum resultado deve produzir decisões automáticas, punições ou exclusão de oportunidades. Todo alerta precisa de avaliação humana.

## Principais melhorias desta versão

- target explícito e configurável;
- pares longitudinais preservam `RA`, `Ano_N` e `Ano_N1`;
- avaliação principal temporal: 2022 para 2023 no treino e 2023 para 2024 no teste;
- comparação com baseline de prevalência e persistência;
- comparação entre modelo completo e modelo acionável;
- feature engineering dentro do `Pipeline`;
- tratamento de ausentes sem excluir silenciosamente todos os registros incompletos;
- calibração de probabilidades;
- threshold escolhido apenas dentro do conjunto de treino;
- métricas ROC-AUC, PR-AUC, recall, precisão, F1, Brier e matriz de confusão;
- validação de entradas e preservação de erros no Streamlit;
- importância global nomeada corretamente;
- versões e metadados do treinamento registrados.

## Resultado validado com a base PEDE 2024

O artefato incluído neste pacote foi treinado no modo `future_defasagem`, com o
perfil `complete`. A avaliação temporal produziu:

| Métrica | Resultado |
|---|---:|
| ROC-AUC | 0,8484 |
| PR-AUC | 0,7900 |
| Recall | 77,60% |
| Precisão | 63,56% |
| F1-score | 0,6988 |
| Brier score | 0,1719 |

Foram utilizados 600 pares de 2022→2023 no treino e 765 pares de 2023→2024 no
teste temporal. O threshold operacional foi 0,4934. Esses valores devem ser
recalculados sempre que a base ou a definição do target mudar.

## Definições possíveis do target

O treinamento aceita duas interpretações:

### 1. `future_defasagem` - padrão

Prevê se o aluno apresentará `Defasagem_N1 <= -1` no ciclo seguinte, incluindo alunos que já estavam defasados.

Pergunta de negócio:

> Qual é a probabilidade de o aluno apresentar defasagem no próximo ciclo?

### 2. `new_defasagem`

Considera somente alunos com `Defasagem_N >= 0` e prevê se eles entrarão em defasagem no ciclo seguinte.

Pergunta de negócio:

> Entre os alunos atualmente adequados, quem possui maior risco de entrar em defasagem?

Essa modalidade pode gerar uma amostra menor. Verifique a quantidade de positivos antes de adotá-la.

## Perfis de modelo

| Perfil | Informações utilizadas | Objetivo |
|---|---|---|
| `complete` | Inclui IAN e Defasagem atual | Maior capacidade preditiva |
| `actionable` | Exclui IAN e Defasagem atual | Avaliar sinais anteriores acadêmicos, emocionais e de engajamento |

O script sempre avalia os dois perfis. O argumento `--model-profile` define qual deles será salvo para produção.

## Estrutura

```text
.
├── analise_pede_datathon.ipynb
├── app.py
├── model_utils.py
├── train_model.py
├── requirements.txt
├── requirements-dev.txt
├── devcontainer.json
├── tests/
│   ├── test_model_utils.py
│   └── test_training_logic.py
└── model_risco_defasagem.pkl       # gerado após o treinamento
```

O notebook `analise_pede_datathon.ipynb` contém o estudo exploratório das três
abas, as respostas às 11 perguntas do Datathon e a avaliação temporal alinhada
ao `train_model.py`.

## Preparação do ambiente

Recomendado: Python 3.11.

O deploy também é compatível com Python 3.14. As versões de Streamlit e pandas
foram selecionadas com pacotes binários prontos para essa versão, evitando a
compilação demorada de dependências no Community Cloud.

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Instalação:

```bash
pip install -r requirements.txt
```

Para executar os testes:

```bash
pip install -r requirements-dev.txt
pytest -q
```

## Treinamento

Modelo completo para presença de defasagem futura:

```bash
python train_model.py \
  --data BASE_DE_DADOS_PEDE_2024_-_DATATHON.xlsx \
  --target-mode future_defasagem \
  --model-profile complete
```

Modelo acionável para nova entrada em defasagem:

```bash
python train_model.py \
  --data BASE_DE_DADOS_PEDE_2024_-_DATATHON.xlsx \
  --target-mode new_defasagem \
  --model-profile actionable
```

Arquivos gerados:

- `model_risco_defasagem.pkl`: Pipeline treinado e metadados necessários pelo app;
- `model_risco_defasagem.metadata.json`: versão legível das métricas e configurações.

## Estratégia de validação

O modelo é avaliado simulando o uso futuro:

- treino: indicadores de 2022 para prever 2023;
- teste intocado: indicadores de 2023 para prever 2024.

O threshold operacional é escolhido em uma divisão interna do conjunto de treino, buscando atingir o recall definido em `--recall-target`. O teste de 2024 não participa da seleção do threshold.

O modelo final de produção é treinado com todos os pares somente depois que a avaliação temporal foi concluída.

## Execução do Streamlit

```bash
streamlit run app.py
```

O arquivo `model_risco_defasagem.pkl` e o módulo `model_utils.py` precisam estar na mesma pasta do `app.py`.

## Processamento em lote

O app aceita CSV e XLSX. No resultado:

- registros válidos recebem probabilidade e classificação;
- células numéricas ausentes são imputadas pelo mesmo Pipeline do treinamento e geram aviso;
- registros inválidos continuam no arquivo;
- `status_processamento` informa se a linha foi avaliada;
- `motivo_erro` descreve campos ausentes, não numéricos, fora da faixa ou RA duplicado.
- `avisos_processamento` registra imputações e pequenos ajustes de arredondamento.

Antes do upload, a própria tela apresenta o nome exato, significado, faixa e
regra de preenchimento de cada campo necessário. O template baixado pelo app já
contém todas as colunas exigidas.

## Interpretação

As faixas operacionais são derivadas do threshold salvo durante o treinamento:

- baixo - rotina;
- monitoramento;
- prioritário.

O gráfico de importância utiliza permutação no teste temporal e representa comportamento global do modelo. Ele não explica uma predição individual.

## Privacidade e uso responsável

- não publicar nomes, CPF ou identificadores diretos;
- utilizar RA anonimizado ou pseudonimizado;
- restringir o uso de dados psicossociais;
- manter supervisão pedagógica e psicossocial;
- acompanhar desempenho por fase e outros grupos relevantes;
- retreinar e revalidar o modelo a cada novo ciclo.

## Entregáveis ainda dependentes do grupo

- adicionar link do Streamlit Community Cloud;
- adicionar apresentação gerencial;
- adicionar vídeo de até cinco minutos;
- preencher integrantes e responsabilidades.
