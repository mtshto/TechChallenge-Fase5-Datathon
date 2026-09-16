# Correções da probabilidade calibrada

Esta versão preserva o modelo e as funcionalidades existentes e melhora a
segurança de interpretação das previsões.

## Alterações realizadas

- A probabilidade da classe `risco = 1` agora é localizada pelo valor da classe,
  sem assumir que ela sempre ocupa a segunda coluna de `predict_proba`.
- A predição individual mostra o threshold operacional e a diferença entre a
  probabilidade calculada e o limite de priorização.
- Valores dentro da escala permitida, mas fora da faixa observada no treinamento,
  geram um alerta de extrapolação e menor confiabilidade.
- O processamento em lote registra esses alertas em
  `avisos_processamento` e marca a linha como `válido com alerta`.
- A interface esclarece que zero é uma nota informada. Valor ausente deve ser
  enviado como célula vazia no arquivo em lote, acionando a imputação do Pipeline.
- Foram adicionados testes para a seleção da classe positiva e para a detecção de
  entradas fora da distribuição observada.

## Caso usado para validação

No cenário com todos os indicadores em zero, `Defasagem_N = -1` e
`Fase_num = 2`, o modelo mantém a probabilidade calibrada de aproximadamente
58,6%. Esse valor está acima do threshold prioritário de 49,34%, mas seis
indicadores estão abaixo do mínimo observado no treinamento. O app agora exibe
as duas informações, evitando apresentar essa extrapolação como uma estimativa
de confiabilidade normal.

## Validação técnica

- 11 testes automatizados aprovados;
- arquivos Python compilados sem erro;
- aplicação Streamlit iniciada corretamente em modo headless.
