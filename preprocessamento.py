import pandas as pd

def main():
    # Carregar o dataset
    df = pd.read_csv('files/dataset.csv')

    # 1. Remover filmes com vote_count = 0
    df = df[df['vote_count'] > 0]

    # 2. Removendo filmes sem gênero
    df = df.dropna(subset=['genres'])

    # 3. Preencher valores ausentes com a média (colunas numéricas)
    colunas_numericas = [
        'vote_average', 'vote_count', 'revenue',
        'runtime', 'budget', 'popularity'
    ]
    df[colunas_numericas] = df[colunas_numericas].fillna(df[colunas_numericas].mean())

    # 4. Criar coluna target (Bom = 1, Ruim = 0)
    df['bom_ruim'] = df['vote_average'].apply(lambda x: 1 if x >= 6.0 else 0)

    # 5. Processar gêneros em variáveis dummy
    genres_split = df['genres'].str.split(', ')
    genres_dummies = pd.get_dummies(genres_split.explode()).groupby(level=0).max()
    df = pd.concat([df, genres_dummies], axis=1)

    # 6. Remover colunas não relevantes
    colunas_para_remover = [
        'id', 'title', 'status', 'imdb_id', 'overview',
        'tagline', 'genres', 'spoken_languages', 'keywords'
    ]
    df = df.drop(columns=colunas_para_remover)

    # 7. Dividir 'production_companies' em três colunas
    companies_split = df['production_companies'].str.split(', ', expand=True)
    df['prod_company_1'] = companies_split[0]
    df['prod_company_2'] = companies_split[1]
    df['prod_company_3'] = companies_split[2]
    df = df.drop(columns=['production_companies'])

    # 8. Dividir 'production_countries' em três colunas
    countries_split = df['production_countries'].str.split(', ', expand=True)
    df['prod_country_1'] = countries_split[0]
    df['prod_country_2'] = countries_split[1]
    df['prod_country_3'] = countries_split[2]
    df = df.drop(columns=['production_countries'])

    df.to_csv('files/dataset_preprocessado.csv', index=False)
    print("Dataset pré-processado salvo!")

if __name__ == '__main__':
    main()