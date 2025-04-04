# Trabalho / Prova IA
3 algoritmos de aprendizado de máquina classificação. 

## Dataset
O dataset segue o seguinte modelo:

Com um total de 291.445 filmes
Percentual de filmes bons: 47.8%
Percentual de filmes ruins: 52.2%

## Pré-processamento
Processos realizados:
- Foram removidos os filmes sem votos contabilizados;
- Foram removidos os filmes que não tem nenhum gênero especificado;
- Foram preenchidos os campos númericos vazios com a média dos outros;
- Foi criada a coluna target "bom_ruim", o filme é julgado bom quando a média é maior ou igual a 6.0
- Foram cridas colunas de gêneros (mult_label)
- Foram removidas algumas colunas julgadas desnecessárias (_'id', 'title', 'status', 'imdb_id', 'overview', 'tagline', 'genres', 'spoken_languages', 'keywords'_);
- A coluna production_companies foi dividida em três, usando as três (ou menos) primeiras empresas listadas;
- A coluna production_countries foi dividida em três, usando os três (ou menos) primeiros países listados;