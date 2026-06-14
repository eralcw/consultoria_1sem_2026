# Relatorio de Consultoria - Predicao de Peso Medio em Lotes Avicolas

## 1. Identificacao do projeto

- Tema: consultoria analitica para predicao de `Peso Medio (kg)` em lotes avicolas.
- Base utilizada: `data/raw/Base_Dados_Modelos_peso.xlsx`.
- Variavel resposta: coluna AN da planilha original, correspondente a `Peso Medio (kg)`.
- Objetivo pratico: construir um modelo que ajude a empresa a entender quais fatores mais se relacionam ao peso medio final e permita gerar predicoes para novos lotes.

---

## 2. Resumo executivo

Este trabalho desenvolveu uma analise exploratoria e um modelo supervisionado de regressao para prever o `Peso Medio (kg)` de lotes avicolas a partir das demais variaveis disponiveis na base. Foram feitas etapas de limpeza, padronizacao, criacao de variaveis derivadas e tratamento automatico das variaveis categoricas por meio de dummies.

Foram comparados cinco modelos de regressao: Regressao Linear, Ridge, Lasso, Regressao Bayesiana e KNN Regressor. O melhor desempenho foi obtido pelo modelo `KNN`, com media de `R2 = 0.796`, `RMSE = 0.125` e `MAE = 0.097` em validacao cruzada com 5 folds.

O resultado indica que o comportamento da variavel resposta nao e puramente linear, e que relacoes locais entre lotes semelhantes sao melhor capturadas pelo KNN do que por modelos lineares tradicionais.

---

## 3. Problema de negocio

A empresa deseja compreender melhor os fatores associados ao peso medio final dos lotes e, ao mesmo tempo, dispor de uma ferramenta capaz de estimar esse valor para novos dados operacionais.

Em um contexto de consultoria, isso e relevante porque:

- o peso medio final impacta produtividade e desempenho economico;
- a previsao pode apoiar decisoes operacionais e de manejo;
- a comparacao entre variaveis ajuda a identificar fatores potencialmente importantes para monitoramento.

---

## 4. Objetivo da consultoria

Os objetivos deste trabalho foram:

1. analisar a qualidade e a estrutura da base de dados;
2. definir a variavel resposta correta;
3. tratar os dados para modelagem supervisionada;
4. comparar diferentes algoritmos de regressao;
5. selecionar o modelo com melhor desempenho;
6. disponibilizar um fluxo de predicao reutilizavel para novos arquivos.

---

## 5. Descricao da base

A base possui informacoes sobre lotes avicolas, contendo variaveis produtivas, de manejo, alimentacao, mortalidade, datas, origem e informacoes operacionais.

Exemplos de variaveis presentes:

- produtor
- regiao
- tecnico
- tipo de instalacao
- nutricao
- genetica
- origem
- peso do pintinho
- aves alojadas
- aves abatidas
- idade media
- pesos em diferentes semanas
- percentuais de mortalidade
- componentes de racao

A analise de qualidade indicou uma base bastante completa, com apenas um valor ausente relevante em `vazio_dias` e baixa incidencia de problemas estruturais.

---

## 6. Limpeza e preparacao dos dados

O tratamento dos dados foi implementado em `src/features/wrangle.py`.

As principais etapas foram:

1. padronizacao dos nomes das colunas;
2. remocao de duplicatas;
3. conversao das colunas de data;
4. criacao de variaveis numericas derivadas das datas;
5. criacao de indicadores derivados uteis para modelagem.

Variaveis derivadas criadas:

- `duracao_ciclo_dias`
- `data_alojamento_ano`, `data_alojamento_mes`, `data_alojamento_dia`
- `data_abate_ano`, `data_abate_mes`, `data_abate_dia`
- `racao_total_por_ave`
- `sobra_racao_pct`
- `taxa_abate`
- `mortalidade_total_pct`

As variaveis categoricas foram transformadas em dummies automaticamente pelo pipeline de preprocessamento.

---

## 7. Cuidados com vazamento de informacao

Um ponto importante da consultoria foi identificar colunas que nao deveriam entrar na modelagem por causarem vazamento da resposta. Entre elas:

- `peso_abatido_kg`
- `aves_abatidas`
- `peso_medio_calculado_kg`
- `ganho_peso_kg`

Essas variaveis foram removidas do conjunto de preditoras porque carregam informacao diretamente ligada ao alvo ou derivada dele. Se fossem mantidas, o modelo teria desempenho artificialmente alto, mas sem validade pratica para generalizacao.

---

## 8. Analise exploratoria dos dados

A EDA foi executada em `src/evaluation/run_eda.py` e gerou figuras e tabelas em `reports/figures` e `reports/tables`.

Os principais pontos observados foram:

1. a variavel resposta `peso_medio_kg` apresenta distribuicao continua e relativamente concentrada;
2. `racao_total_por_ave`, `idade_media_dias` e variaveis de crescimento se mostraram relacionadas ao peso medio;
3. a base possui boa cobertura e poucas falhas de preenchimento;
4. havia sinais de relacoes nao estritamente lineares entre preditoras e alvo.

Graficos recomendados para a apresentacao:

- `target_distribution.png`
- `correlation_heatmap.png`
- `target_vs_feed_per_bird.png`
- `target_by_region.png`
- `target_vs_mortality.png`

---

## 9. Modelos avaliados

Foram comparados os seguintes algoritmos:

1. Regressao Linear
2. Ridge
3. Lasso
4. Regressao Bayesiana
5. K-Nearest Neighbors Regressor (KNN)

A avaliacao foi feita com validacao cruzada de 5 folds, usando as metricas:

- `R2`
- `RMSE`
- `MAE`

---

## 10. O que sao os 5 folds

Os 5 folds representam 5 divisoes diferentes da base para validacao.

Em cada rodada:

1. o modelo treina em aproximadamente 80% dos dados;
2. testa nos 20% restantes;
3. esse processo se repete 5 vezes, mudando o grupo de teste.

Isso torna a avaliacao mais confiavel do que usar apenas uma unica divisao treino/teste.

---

## 11. Resultados da comparacao de modelos

Os resultados medios observados foram:

| Modelo | R2 medio | RMSE medio | MAE medio |
|---|---:|---:|---:|
| KNN | 0.796 | 0.125 | 0.097 |
| Lasso | -28.649 | 0.717 | 0.075 |
| Regressao Linear | -336.566 | 2.635 | 0.128 |
| Ridge | -451.153 | 3.003 | 0.138 |
| Regressao Bayesiana | -517.272 | 3.204 | 0.144 |

O melhor modelo foi o `KNN`.

---

## 12. Resultado dos 5 folds do modelo vencedor

| Fold | R2 | RMSE | MAE |
|---|---:|---:|---:|
| 1 | 0.7967 | 0.1268 | 0.0997 |
| 2 | 0.8019 | 0.1253 | 0.0972 |
| 3 | 0.7901 | 0.1287 | 0.1000 |
| 4 | 0.7995 | 0.1213 | 0.0938 |
| 5 | 0.7924 | 0.1210 | 0.0935 |

Interpretacao:

- o desempenho foi estavel entre os folds;
- nao houve grande oscilacao entre as rodadas;
- isso sugere que o modelo apresenta consistencia e boa capacidade de generalizacao dentro da base disponivel.

---

## 13. Por que o KNN foi o melhor modelo

O KNN apresentou melhor desempenho porque conseguiu capturar melhor a estrutura local dos dados.

Em termos prticos:

1. os modelos lineares tentam explicar toda a base com uma relacao global unica;
2. o KNN prediz com base em observacoes semelhantes;
3. como os dados parecem nao seguir uma relacao linear simples, o KNN se ajustou melhor.

Portanto, a principal conclusao tecnica e que o peso medio final parece depender de padroes locais de semelhanca entre lotes, e nao apenas de uma formula linear global.

---

## 14. Entregaveis do projeto

Os principais entregaveis gerados foram:

- limpeza e preparacao em `src/features/wrangle.py`
- EDA em `src/evaluation/run_eda.py`
- comparacao de modelos em `src/models/train_and_compare.py`
- predicao em `src/models/predict.py`
- modelo salvo em `models/best_model.joblib`
- predicoes em `data/processed/predicoes_teste.csv`
- resposta formal em `docs/resposta_modelagem.md`

---

## 15. Como fazer novas predicoes

O fluxo de predicao ficou pronto para reutilizacao.

Comando:

```powershell
.\.venv\Scripts\python.exe -m src.models.predict data\raw\Base_Dados_Modelos_peso.xlsx --output data\processed\predicoes.csv
```

Esse comando:

1. le o arquivo informado;
2. aplica a mesma limpeza usada no treino;
3. carrega o melhor modelo salvo;
4. gera a coluna `predicted_peso_medio_kg`;
5. salva o resultado em CSV.

---

## 16. Recomendacoes para a empresa

Com base neste estudo, as recomendacoes de consultoria sao:

1. monitorar continuamente variaveis operacionais relacionadas a consumo de racao, idade media e mortalidade;
2. manter historico organizado e padronizado dos lotes;
3. utilizar o modelo preditivo como apoio a decisao, e nao como substituto do julgamento tecnico;
4. atualizar o modelo periodicamente com novos dados;
5. em estudos futuros, testar um cenario de previsao antecipada, removendo variaveis muito proximas do final do ciclo.

---

## 17. Limitacoes do trabalho

Este trabalho possui algumas limitacoes:

1. o modelo foi avaliado com validacao cruzada interna, e nao com uma base externa independente;
2. o KNN oferece bom desempenho, mas menor interpretabilidade do que modelos lineares;
3. os resultados dependem da qualidade e representatividade da base atual;
4. o modelo identifica padroes associativos, nao causalidade.

---

## 18. Conclusao

O projeto cumpriu o objetivo de construir uma analise de consultoria baseada em dados para predicao de `Peso Medio (kg)` em lotes avicolas. A base foi tratada adequadamente, as variaveis categoricas foram convertidas em dummies, foram comparados diferentes modelos de regressao e o melhor desempenho foi obtido pelo `KNN`.

A consultoria mostra que a empresa pode utilizar os dados historicos para apoiar decisoes e gerar estimativas consistentes de peso medio final, desde que mantenha a qualidade do cadastro das informacoes e atualize periodicamente o processo de modelagem.

---

## 19. Roteiro sugerido para a apresentacao do grupo

### Slide 1 - Tema
- titulo do projeto
- nome do grupo
- empresa ou contexto da consultoria

### Slide 2 - Problema
- o que a empresa quer resolver
- por que prever peso medio e importante

### Slide 3 - Base de dados
- origem da base
- quantidade de variaveis
- variavel resposta

### Slide 4 - Tratamento dos dados
- limpeza
- dummies
- remocao de vazamento

### Slide 5 - EDA
- mostrar 2 ou 3 graficos principais
- comentar padroes encontrados

### Slide 6 - Modelos testados
- listar os 5 modelos
- explicar em uma frase o uso de 5 folds

### Slide 7 - Resultados
- tabela com metricas medias
- destacar KNN como melhor modelo

### Slide 8 - Interpretacao
- explicar por que o KNN foi melhor
- comentar o significado pratico das metricas

### Slide 9 - Predicao pronta
- mostrar que o modelo foi salvo
- mostrar que gera previsoes para novos dados

### Slide 10 - Recomendacoes finais
- o que a empresa pode fazer com o resultado
- limitacoes e proximos passos
