import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
import joblib
import os

# Configurações globais
RANDOM_STATE = 42
TEST_SIZE = 0.2
OUTPUT_PATH = 'output/knn/'
MODEL_PATH = 'weights/knn/'


def load_data(file_path='files/dataset_preprocessado.csv'):
    """Carrega e prepara os dados"""
    df = pd.read_csv(file_path)

    cols_to_drop = ['release_date', 'original_title', 'vote_average',  # vote_average para não dar data leakage
                    'prod_company_1', 'prod_company_2', 'prod_company_3',
                    'prod_country_1', 'prod_country_2', 'prod_country_3', ]
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

    # Converter booleanos
    bool_cols = df.select_dtypes(include=['bool']).columns
    df[bool_cols] = df[bool_cols].astype(int)

    return df


def preprocess_data(df):
    """Pré-processamento padrão"""
    X = df.drop(columns=['bom_ruim'])
    y = df['bom_ruim']

    # Normalização
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.select_dtypes(include=['number']))

    # Seleção de features
    selector = SelectKBest(f_classif, k=15)
    X_selected = selector.fit_transform(X_scaled, y)

    return X_selected, y, scaler, selector


def train_and_evaluate(X, y):
    """Treina e avalia o modelo com GridSearch"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # Grid de parâmetros
    param_grid = {
        'n_neighbors': np.arange(3, 30, 2),
        'weights': ['uniform'],
        'metric': ['euclidean']
    }

    grid = GridSearchCV(
        KNeighborsClassifier(),
        param_grid,
        cv=5,
        n_jobs=-1,
        scoring='accuracy'
    )
    grid.fit(X_train, y_train)

    # Melhor modelo
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)

    # Salvar métricas
    save_metrics(y_test, y_pred, grid)
    save_artifacts(best_model)

    return best_model


def save_metrics(y_test, y_pred, grid):
    """Salva métricas e gráficos"""
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    # Matriz de confusão
    plt.figure(figsize=(6, 6))
    sns.heatmap(confusion_matrix(y_test, y_pred),
                annot=True, fmt='d', cmap='Blues')
    plt.title('Matriz de Confusão - KNN')
    plt.savefig(f'{OUTPUT_PATH}confusion_matrix.png')
    plt.close()

    # Relatório de classificação
    with open(f'{OUTPUT_PATH}classification_report.txt', 'w') as f:
        f.write(classification_report(y_test, y_pred))
        f.write(f"\nMelhores parâmetros: {grid.best_params_}")


def save_artifacts(model):
    """Salva modelo e pré-processadores"""
    os.makedirs(MODEL_PATH, exist_ok=True)
    joblib.dump(model, f'{MODEL_PATH}knn_model.pkl')


if __name__ == "__main__":
    df = load_data()
    print("Tamanho total do dataset:", len(df))
    X, y, scaler, selector = preprocess_data(df)
    model = train_and_evaluate(X, y)
    print("Processo concluído. Artefatos salvos nas pastas output/ e models/")
