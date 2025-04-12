# Trabalho / Prova IA

3 algoritmos de aprendizado de máquina para classificação.

## Dataset

O dataset segue o seguinte modelo:

Com um total de 73.277 filmes
Percentual de filmes bons:  50%
Percentual de filmes ruins: 50%

## Pré-processamento

Processos realizados:

- Foram removidos os filmes sem votos contabilizados;
- Foram removidos os filmes que não tem nenhum gênero especificado;
- Foram preenchidos os campos númericos vazios com a média dos outros;
- Foi criada a coluna target "bom_ruim", o filme é julgado bom quando a média é maior ou igual a mediana dos votos
- Foram cridas colunas de gêneros (mult_label)
- Foram removidas algumas colunas julgadas desnecessárias (_'id', 'title', 'status', 'imdb_id', 'overview', 'tagline', '
  genres', 'spoken_languages', 'keywords'_);
- A coluna production_companies foi dividida em três, usando as três (ou menos) primeiras empresas listadas;
- A coluna production_countries foi dividida em três, usando os três (ou menos) primeiros países listados;

## Resultados

|    Modelo     | Acurácia (%) |                                               Melhores hiperparâmetros                                               |
|:-------------:|:------------:|:--------------------------------------------------------------------------------------------------------------------:|
|      KNN      |      68      |                      {'metric': 'euclidean', 'n_neighbors': np.int64(39), 'weights': 'uniform'}                      |
| Random Forest |      70      |                            {'max_depth': 10, 'max_features': 'sqrt', 'n_estimators': 100}                            |
|    XGBoost    |      72      | {'colsample_bytree': 0.8, 'gamma': 0.1, 'learning_rate': 0.1, 'max_depth': 7, 'n_estimators': 150, 'subsample': 0.8} |

## Respostas

### 1 (1.0 ponto) Encontre no Kaggle (https://www.kaggle.com) ou repositório equivalente na internet uma base de dados que possa ser explorada utilizando: (a) Métodos de Aprendizado de Máquina para Classificação, ou ainda (b) Métodos de Aprendizado de Máquina para Regressão. Insira o link da base escolhida no arquivo público da turma postado no Classroom. Explicar os atributos de entrada, o atributo-meta (saída), e sua motivação para trabalhar com o problema de classificação (ou regressão) escolhido.

Utilizado o dataset [Full IMDb Movies Data](https://www.kaggle.com/datasets/anandshaw2001/imdb-data) para *
*classificação**
de filmes como bom (média de votos >= 6.3) ou ruim (média de votos < 6.3). Os atibutos de entrada são:

- **vote_average**: Média de votos dos usuários;
- **vote_count**: Total de votos recebidos em um filme;
- **revenue**: Ganhos totais gerados pelo filme;
- **runtime**: Duração do filme em minutos;
- **adult**: Indica se é um filme adulto (booleano);
- **budget**: Orçamento do filme;
- **popularity**: Pontuação de popularidade do filme;
- **ROI**: Retorno do investimento do filme;
- **profit**: Lucro do filme;
- **Action**: Filme de ação (booleano);
- **Adventure**: Filme de aventura (booleano);
- **Animation**: Filme de animação (booleano);
- **Comedy**: Filme de comédia (booleano);
- **Crime**: Filme de crime (booleano);
- **Documentary**: Filme documentário (booleano);
- **Drama**: Filme de drama (booleano);
- **Family**: Filme família (booleano);
- **Fantasy**: Filme de fantasia (booleano);
- **Hystory**: Filme de história (booleano);
- **Horror**: Filme de horror (booleano);
- **Music**: Filme musical (booleano);
- **Mystery**: Filme de mistério (booleano);
- **Romance**: Filme de romance (booleano);
- **Science Fiction**: Filme de ficção científica (booleano);
- **TV Movie**: Série de TV (booleano);
- **Thriller**: Filme de terror (booleano);
- **War**: Filme de guerra (booleano);
- **Western**: Filme de velho-oeste (booleano);
- **bom_ruim**: Coluna target (vote_average >= 6.3 = 1 // vote_average < 6.3 = 0);

O dataset me chamou atenção por conter ao todo mais de um milhão de registros (que após o pré-processamento)
a quantidade foi reduzida para 73.277 registros. A ideia de trabalhar com a média de votos foi o melhor caminho
para deixar o dataset balanceado para classificar (51,7% ruim // 48,8% bom)

### 2. (6.0 pontos) Empregar três algoritmos de Aprendizado de Máquina (AM) nos conjuntos de dados selecionados na Questão 1 (algoritmos vistos ou não vistos em aula, à sua escolha). Siga as instruções:

Algoritmos selecionados KNN (_k-nearest neighbors_), Random Forest e XGBoost (_eXtreme Gradient Boosting_)

#### a) Conduza as etapas necessárias de pré-processamento do dataset escolhido visando “preparar o dado”para a aplicação dos algoritmos de AM. Forneça detalhes dos processos de pré-processamento que utilizou.

Processos realizados:

- Foram removidos os filmes com poucos votos contabilizados;
- Foram removidos os filmes que não tem nenhum gênero especificado;
- Foram preenchidos os campos númericos vazios (ou valor = 0) com a média dos outros;
- Foi criada a coluna ROI (retorno de investimento) = renda dividido pelo orçamento
- Foi criada a coluna profit (lucro) = renda menos orçamento
- Foram cridas colunas de gêneros (One-Hot Encoding)
- Foram removidas colunas não numéricas ou irrelevantes (como por exemplo o ID do filme);
- Foi criada a coluna target "bom_ruim", o filme é julgado bom quando a média é maior ou igual a 6.232 (mediana dos
  votos)
  O arquivo é então salvo como um csv pré-processado.

O arquivo então é carregado novamente durante a execução dos algoritmos. Primeiramente a coluna "vote_average" é
removida
para evitar data leakage, em seguida são realizadas etapas adicionais específicas para cada modelo para otimizar o
desempenho:

- **Para o modelo KNN:** é feita a normalização dos dados. Todas as colunas numéricas são padronizadas usando
  StandardScaler. Isso transforma os dados para ter média 0 e desvio padrão 1, evitando que features com escalas maiores
  dominem o cálculo de distâncias. Além disso também é realizada a extração de features usando SelectKBest com o teste
  ANOVA (f_classif) para selecionar as 15 features mais relevantes para a classificação.
- **Para o modelos XGB:** é feito One-Hot Encoding de colunas categóricas (embora essas colunas não existam mais depois
  do dataset set tratado anteriormente é uma redundância que age como camada de segurança
- **Para o modelo RF:** nenhuma transformação adicional é necessária

#### b) Aplique os três algoritmos nos dados pós-processados. Varie os hiperparametros dos algoritmos, nos casos pertinentes, de forma a melhor ajustar os modelos, melhorando assim os resultados.

Para todos os modelos foi feito cross_validation com 5 folds, a seguir estão os hiperparâmetros de cada modelo

##### KNN

- **n_neighbors (Número de vizinhos):** Foi testado valores ímpares de 3 a 49, o melhor encontrado foi 39
- **weights (Pesos):** Foi testados com pesos 'uniform' e 'distance', o melhor encontrado foi 'uniform'
- **metric (Métrica de distância):** Foi testado com 'euclidean', 'manhattan' e 'cosine', o melhor encontrado foi 'euclidean'

##### Random Forest

- **n_estimators (Número de árvores de decisão na floresta):** Testado com valores 50 e 100. O melhor encontrado foi 100.
- **max_depth (Profundidade máxima das árvores):** Testado com None (sem limite) e 10. O melhor encontrado foi 10
- **max_features (Número de features por divisão):** Testado com 'sqrt' (raiz quadrada do total de features) e 0.5 (50%
  das features). O melhor resultado foi 'sqrt'

##### XGBoost

- **n_estimators (Número de árvores):** Testado com 50, 100, e 150. O melhor foi 150
- **max_depth (Profundidade máxima das árvores):** Testado com 3, 5, e 7. O valor ótimo foi 7
- **learning_rate (Taxa de aprendizado):** Testado com 0.01, 0.1, e 0.2. O melhor foi 0.1
- **subsample (Fração de amostras por árvore):** Testado com 0.8 e 1.0 (todas as amostras). O melhor foi 0.8
- **gamma (Redução mínima de perda para split):** Testado com 0, 0.1, e 0.2. O melhor foi 0.1
- **colsample_bytree (Fração de features por árvore):** Testado com 0.8 e 1.0. O melhor foi 0.8

#### c) Para quem optou por trabalhar com a aplicação de Classificação: (i) apresente os resultados obtidos utilizando ao menos duas métricas de avaliação; exiba também a Matriz de Confusão. (ii) Aplique alguma(s) ferramenta(s) de visualização (escolha livre) para analisar os dados. (iii) Compare os resultados obtidos pelos três algoritmos.
##### (i) Resultados das Métricas e Matriz de Confusão
Foram utilizadas as métricas precisão, recall, F1-score e acurácia para avaliar os modelos. Abaixo estão os resultados detalhados:

###### KNN
- **Acurácia:** 68%
- **Precisão (Classe 0):** 0.68 **| Precisão (Classe 1):** 0.68
- **Recall (Classe 0):** 0.73 **| Recall (Classe 1):** 0.63
- **F1-score (Macro):** 0.68
- **Matriz de Confusão:** 


###### Random Forest
- Acurácia: 70%
- **Precisão (Classe 0):** 0.69 **| Precisão (Classe 1):** 0.71
- **Recall (Classe 0):** 0.75 **| Recall (Classe 1):** 0.64
- **F1-score (Macro):** 0.69
- **Matriz de Confusão:**

###### XGBoost
- **Acurácia:** 72%
- **Precisão (Classe 0):** 0.71 **| Precisão (Classe 1):** 0.72
- **Recall (Classe 0):** 0.76 **| Recall (Classe 1):** 0.67
- **F1-score (Macro):** 0.72
- **Matriz de Confusão:**

##### (ii) Visualização
Foram aplicadas as seguintes ferramentas de visualização:
- **Matriz de Correlação:** Para identificar relações entre variáveis como budget, revenue e popularity com a classe bom_ruim.
- **Gráfico de Importância de Features (Random Forest/XGBoost):** Destacando as variáveis mais relevantes para a classificação (ex: ROI, popularity, budget).
- **Curva ROC:** Comparando a área sob a curva (AUC) dos três modelos.

##### (iii) Comparação dos Algoritmos
- **KNN:** Teve o pior desempenho (acurácia 68%), possivelmente devido à sensibilidade a outliers e alta dimensionalidade. A normalização e seleção de features melhoraram parcialmente o modelo.
- **Random Forest:** Acurácia intermediária (70%), com melhor balanceamento entre precisão e recall.
- **XGBoost:** Obteve a melhor performance (72%).