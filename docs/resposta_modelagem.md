# Resposta formal sobre a modelagem

Prezados,

Foi realizada a modelagem preditiva utilizando a coluna AN do arquivo, correspondente a `Peso Medio (kg)`, como variavel resposta.

Foram consideradas como variaveis preditoras todas as demais colunas da base que podem ser utilizadas sem gerar vazamento da resposta. As variaveis categoricas foram transformadas em dummies por meio de codificacao one-hot, e foi feita limpeza dos dados com padronizacao dos nomes das colunas, conversao de datas, remocao de duplicatas e criacao de variaveis derivadas relevantes para a modelagem.

Durante a analise, algumas colunas foram identificadas como inadequadas para predicao direta, pois carregam informacao derivada da propria resposta ou muito proxima dela. Esse cuidado foi necessario para evitar um modelo artificialmente bom, mas sem capacidade real de generalizacao.

Foram comparados os seguintes modelos de regressao supervisionada:

- Regressao Linear
- Ridge
- Lasso
- Regressao Bayesiana
- K-Nearest Neighbors Regressor (KNN)

A comparacao foi feita com validacao cruzada de 5 folds e com as metricas `R2`, `RMSE` e `MAE`.

No cenario com todas as variaveis utilizaveis da base, o melhor desempenho foi obtido pelo modelo `KNN`, com aproximadamente:

- `R2 = 0.796`
- `RMSE = 0.125`
- `MAE = 0.097`

Esse resultado indica que o KNN representou melhor a estrutura dos dados do que os modelos lineares, sugerindo que a relacao entre as variaveis explicativas e a resposta nao e puramente linear.

Para realizar novas predicoes a partir do modelo treinado, o fluxo adotado foi:

1. aplicar a mesma limpeza e transformacao ao novo arquivo;
2. carregar o modelo salvo;
3. gerar a coluna de predicao `predicted_peso_medio_kg`.

O modelo final foi salvo em `models/best_model.joblib` e pode ser reutilizado diretamente em novos arquivos com a mesma estrutura.

Atenciosamente.