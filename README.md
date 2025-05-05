# Trabalho / Prova IA

3 algoritmos de aprendizado de máquina para classificação.

## Dataset

O dataset segue o seguinte modelo:

Com um total de 73.277 filmes
Percentual de filmes bons:  50%
Percentual de filmes ruins: 50%

## Pré-processamento

Processos realizados (função transforma_dataset):

- Foram removidos os filmes com poucos votos contabilizados;
- Foram removidos os filmes que não tem nenhum gênero especificado;
- Foram preenchidos os campos númericos vazios (ou valor = 0) com a média dos outros;
- Foi criada a coluna ROI (retorno de investimento) = renda dividido pelo orçamento
- Foi criada a coluna profit (lucro) = renda menos orçamento
- Foram cridas colunas de gêneros (One-Hot Encoding)
- Foram removidas colunas não numéricas ou irrelevantes (como por exemplo o ID do filme);
- Foi criada a coluna target "bom_ruim", o filme é julgado bom quando a média é maior ou igual a 6.232 (mediana dos votos) O arquivo é então salvo como um csv pré-processado.
Vale ressaltar que tentei remover os outliners porém os resultados foram piores do que com todos os outliners

O arquivo então é carregado novamente durante a execução dos algoritmos. Primeiramente a coluna "vote_average" é removida para evitar data leakage, em seguida são realizadas etapas adicionais específicas para cada modelo para otimizar o desempenho:

- Para o modelo KNN: é feita a normalização dos dados. Todas as colunas numéricas são padronizadas usando StandardScaler. Isso transforma os dados para ter média 0 e desvio padrão 1, evitando que features com escalas maiores dominem o cálculo de distâncias. Além disso também é realizada a extração de features usando SelectKBest com o teste ANOVA (f_classif) para selecionar as 15 features mais relevantes para a classificação.
- Para o modelos XGB: é feito One-Hot Encoding de colunas categóricas (embora essas colunas não existam mais depois do dataset set tratado anteriormente é uma redundância que age como camada de segurança
- Para o modelo RF: nenhuma transformação adicional é necessária;

## Resultados

|    Modelo     | Acurácia (%) |                                               Melhores hiperparâmetros                                               |
|:-------------:|:------------:|:--------------------------------------------------------------------------------------------------------------------:|
|      KNN      |      68      |                      {'metric': 'euclidean', 'n_neighbors': np.int64(39), 'weights': 'uniform'}                      |
| Random Forest |      70      |                            {'max_depth': 10, 'max_features': 'sqrt', 'n_estimators': 100}                            |
|    XGBoost    |      72      | {'colsample_bytree': 0.8, 'gamma': 0.1, 'learning_rate': 0.1, 'max_depth': 7, 'n_estimators': 150, 'subsample': 0.8} |
