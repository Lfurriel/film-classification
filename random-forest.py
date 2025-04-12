# random_forest_filmes.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Configurações iniciais
RANDOM_STATE = 42
TEST_SIZE = 0.2


# 1. Carregar os dados
def load_data(file_path):
    df = pd.read_csv(file_path)
    df = df.dropna(axis='rows')
    return df


# 2. Pré-processamento
def preprocess(df):
    # Converter data para ano numérico
    df['release_year'] = pd.to_datetime(df['release_date']).dt.year

    # Remover colunas não numéricas que não serão usadas
    df = df.drop(['release_date', 'original_title', 'prod_company_1',
                  'prod_company_2', 'prod_company_3',
                  'prod_country_1', 'prod_country_2', 'prod_country_3', 'original_language', 'vote_average'],
                 axis=1, errors='ignore')

    # Converter colunas booleanas para numéricas
    bool_cols = df.select_dtypes(include=['bool']).columns
    df[bool_cols] = df[bool_cols].astype(int)

    # Separar features e target
    X = df.drop('bom_ruim', axis=1)
    y = df['bom_ruim']
    return X, y


# 3. Treinar e avaliar o modelo
def train_and_evaluate(X, y):
    # Dividir dados
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # Criar modelo
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
        max_features='sqrt'
    )

    # Treinar
    model.fit(X_train, y_train)

    # Previsões
    y_pred = model.predict(X_test)

    # Métricas
    print(f"Acurácia: {accuracy_score(y_test, y_pred):.2%}")

    # Matriz de confusão
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Matriz de Confusão')
    plt.xlabel('Previsto')
    plt.ylabel('Real')
    plt.savefig('images/random_forest/matriz_confusao.png')
    plt.close()

    # Importância das features
    importances = model.feature_importances_
    indices = np.argsort(importances)[-15:]  # Top 15 features
    plt.figure(figsize=(10, 8))
    plt.title('Importância das Features')
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [X.columns[i] for i in indices])
    plt.savefig('images/random_forest/importancia_features.png')
    plt.close()


# Execução principal
if __name__ == "__main__":
    FILE_PATH = 'files/dataset_preprocessado.csv'

    print("Carregando dados...")
    df = load_data(FILE_PATH)

    print("Pré-processando...")
    X, y = preprocess(df)

    print("Features utilizadas:", list(X.columns))

    print("Treinando modelo...")
    train_and_evaluate(X, y)

    print("Processo concluído! Verifique as imagens geradas.")