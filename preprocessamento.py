import pandas as pd
import numpy as np


def main():
    # Carregar o dataset
    df = pd.read_csv('files/dataset.csv')
    print("Tamanho total do dataset:", len(df))

    # 1. Remover filmes com vote_count <= 10
    df = df[df['vote_count'] > 10]

    # 2. Removendo filmes sem gênero
    df = df.dropna(subset=['genres'])

    # 3. Preencher valores ausentes com a média (colunas numéricas)
    colunas_numericas = [
        'vote_average', 'vote_count', 'revenue',
        'runtime', 'budget', 'popularity'
    ]
    df[colunas_numericas] = df[colunas_numericas].replace(0, np.nan)
    df[colunas_numericas] = df[colunas_numericas].fillna(df[colunas_numericas].mean())

    # 4. Adicionando Return Of Ivestiment
    df['ROI'] = df['revenue'] / (df['budget'] + 1)  # +1 para evitar divisão por zero

    # 5. Adicionando receita
    df['profit'] = df['revenue'] - df['budget']

    # 6. Processar gêneros em variáveis dummy
    genres_split = df['genres'].str.split(', ')
    genres_dummies = pd.get_dummies(genres_split.explode()).groupby(level=0).max()
    df = pd.concat([df, genres_dummies], axis=1)

    # 7. Remover colunas não relevantes
    colunas_para_remover = [
        'id', 'title', 'status', 'imdb_id', 'overview', 'tagline', 'original_language',
        'genres', 'spoken_languages', 'keywords', 'release_date', 'original_title',
        'production_companies', 'production_countries'
    ]
    df = df.drop(columns=colunas_para_remover)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # 8. Criar coluna target (Bom = 1, Ruim = 0)
    limiar = df['vote_average'].median() # 6.232
    df['bom_ruim'] = df['vote_average'].apply(lambda x: 1 if x >= limiar else 0)

    print("\nDistribuição de classes:")
    print(df['bom_ruim'].value_counts(normalize=True))
    print("Após limpeza final:", len(df))

    df.to_csv('files/dataset_preprocessado.csv', index=False)
    print("Dataset pré-processado salvo!")

if __name__ == '__main__':
    main()
