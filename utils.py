import os
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import pandas as pd


def load_data(file_path='files/dataset_preprocessado.csv'):
    df = pd.read_csv(file_path)

    cols_to_drop = ['release_date', 'original_title', 'vote_average',  # vote_average para não dar data leakage
                    'prod_company_1', 'prod_company_2', 'prod_company_3',
                    'prod_country_1', 'prod_country_2', 'prod_country_3', ]
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

    # Converter booleanos
    bool_cols = df.select_dtypes(include=['bool']).columns
    df[bool_cols] = df[bool_cols].astype(int)

    return df


def preprocess_data(df, model):
    X = df.drop(columns=['bom_ruim'])
    y = df['bom_ruim']

    if model == 'KNN':
        numeric_cols = X.select_dtypes(include=['number']).columns
        scaler = StandardScaler()
        X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

        # Processamento adicional para KNN
        selector = SelectKBest(f_classif, k=15)
        X_selected = selector.fit_transform(X, y)

        return X_selected, y
    elif model == 'RF':
        return X, y
    elif model == 'XGB':
        X_processed = pd.get_dummies(X, drop_first=True)
        print("Número de features após transformação:", X_processed.shape[1])
        return X, y, X_processed
    else:
        raise ValueError("Modelo não suportado. Use 'KNN', 'RF' ou 'XGB'")


def save_metrics(y_test, y_pred, grid, output_path):
    os.makedirs(output_path, exist_ok=True)
    plt.figure(figsize=(6, 6))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.title('Matriz de Confusão')
    plt.savefig(f'{output_path}confusion_matrix.png')
    plt.close()
    with open(f'{output_path}classification_report.txt', 'w') as f:
        f.write(classification_report(y_test, y_pred))
        f.write(f"\nMelhores parâmetros: {grid.best_params_}")


def save_artifacts(model, path):
    os.makedirs(path, exist_ok=True)
    joblib.dump(model, f'{path}model.pkl')
