# Consultoria Analítica - Predição de Peso Médio em Lotes Avícolas

Projeto de consultoria analítica para prever o `Peso Medio (kg)` de lotes avícolas a partir de variáveis operacionais, produtivas e de manejo. O foco do trabalho foi construir um fluxo reproduzível de tratamento de dados, análise exploratória, comparação de modelos e geração de predições para novos arquivos.

## Visão Geral

A base principal do projeto está em `data/raw/Base_Dados_Modelos_peso.xlsx`. A variável resposta utilizada na modelagem é `peso_medio_kg` e o modelo final salvo em `models/best_model.joblib` foi o `KNN`, selecionado após comparação com regressões lineares e regulares.

Os principais entregáveis do repositório são:

- limpeza e engenharia de atributos em `src/features/wrangle.py`
- análise exploratória em `src/evaluation/run_eda.py`
- comparação e treinamento de modelos em `src/models/train_and_compare.py`
- geração de predições em `src/models/predict.py`
- gráficos e tabelas em `reports/figures` e `reports/tables`

## Resultados da Modelagem

Foram comparados os modelos Regressão Linear, Ridge, Lasso, Regressão Bayesiana e KNN usando validação cruzada com 5 folds. O melhor desempenho foi obtido pelo `KNN`, com resultados médios próximos de:

- `R2 = 0.796`
- `RMSE = 0.125`
- `MAE = 0.097`

Esses resultados indicam que a relação entre as variáveis explicativas e o peso médio não é puramente linear, e que padrões locais entre lotes semelhantes foram capturados com mais eficiência pelo KNN.

## Estrutura do Projeto

```text
README.md
configs/
data/
	raw/
	processed/
docs/
models/
notebooks/
reports/
	figures/
	tables/
src/
	evaluation/
	features/
	models/
```

## Como Executar

O repositório foi montado para uso com o ambiente local `.venv`. No Windows, prefira chamar o Python diretamente pela pasta do ambiente:

```powershell
.\.venv\Scripts\python.exe -m src.evaluation.run_eda
.\.venv\Scripts\python.exe -m src.models.train_and_compare
```

Para gerar predições em um novo arquivo:

```powershell
.\.venv\Scripts\python.exe -m src.models.predict caminho\para\arquivo.xlsx --sheet Base_Dados
```

O script de predição aceita arquivos `.xlsx`, `.xls` e `.csv`. Se nenhum caminho de saída for informado, o resultado é salvo com o mesmo nome do arquivo de entrada, acrescido de `_predicoes.csv`.

## Fluxo Reproduzível

1. Carregar e padronizar a base com `wrangle.py`.
2. Executar a EDA e gerar tabelas e figuras em `reports/`.
3. Treinar e comparar os modelos em validação cruzada.
4. Salvar o melhor pipeline em `models/best_model.joblib`.
5. Aplicar o pipeline em novos dados com `src/models/predict.py`.

## Cuidados de Modelagem

Algumas colunas foram removidas do conjunto de variáveis explicativas por representarem vazamento de informação da própria resposta, como `peso_abatido_kg`, `aves_abatidas`, `peso_medio_calculado_kg` e `ganho_peso_kg`. Esse cuidado evita um modelo artificialmente bom, mas sem utilidade prática.

## Observações de Ambiente

- Use `.venv\Scripts\python.exe` para executar os scripts no Windows.
- Para instalar pacotes, prefira `.venv\Scripts\python.exe -m pip install <pacote>`.
- No notebook `notebooks/nt.ipynb`, adicione a raiz do projeto ao `sys.path` antes de importar `src.*`.

## Arquivos de Saída

As principais saídas geradas pelo projeto ficam em:

- `reports/figures`
- `reports/tables`
- `models/best_model.joblib`

## Referências

Mais detalhes sobre a análise e a interpretação dos resultados estão em:

- `docs/relatorio_consultoria_grupo.md`
- `docs/resposta_modelagem.md`
