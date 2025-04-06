import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE
from scipy import stats
import joblib
import os
import warnings

# Configurações iniciais
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
plt.style.use('ggplot')
np.random.seed(42)


def load_data():
    """Carrega e prepara os dados"""
    try:
        df = pd.read_csv('files/dataset_preprocessado.csv', encoding='UTF-8')

        # Colunas a remover
        cols_to_drop = ['vote_average', 'original_title', 'release_date']
        df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

        # Converter booleanos para numéricos
        bool_cols = df.select_dtypes(include=['bool']).columns
        df[bool_cols] = df[bool_cols].astype(int)

        # Codificar colunas categóricas
        for col in df.select_dtypes(include=['object']).columns:
            if len(df[col].unique()) <= 20:
                df[col] = LabelEncoder().fit_transform(df[col].astype(str))
            else:
                df = df.drop(columns=[col])

        return df
    except Exception as e:
        print(f"Erro ao carregar dados: {e}")
        exit()


def exploratory_analysis(df):
    """Realiza análise exploratória dos dados"""
    print("\n=== Análise Exploratória ===")

    print("\nDistribuição de classes:")
    print(df['bom_ruim'].value_counts(normalize=True))

    # Correlação apenas com colunas numéricas
    numeric_df = df.select_dtypes(include=['number'])
    if 'bom_ruim' in numeric_df.columns:
        plt.figure(figsize=(12, 6))
        corr = numeric_df.corr()['bom_ruim'].sort_values()
        corr.drop('bom_ruim', errors='ignore').plot.barh()
        plt.title('Correlação com a variável target')
        plt.tight_layout()
        plt.show()


def feature_engineering(df):
    """Realiza engenharia de features"""
    print("\n=== Engenharia de Features ===")

    if all(col in df.columns for col in ['revenue', 'budget']):
        df['ROI'] = np.where(df['budget'] > 0, df['revenue'] / df['budget'], np.nan)
        df['profit'] = df['revenue'] - df['budget']

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df.dropna()


def preprocess_data(df):
    """Pré-processa os dados"""
    print("\n=== Pré-processamento ===")

    X = df.drop(columns=['bom_ruim'])
    y = df['bom_ruim']

    # Normalização
    numeric_cols = X.select_dtypes(include=['number']).columns
    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    # Seleção de features
    selector = SelectKBest(f_classif, k=min(15, len(numeric_cols)))
    X_selected = selector.fit_transform(X[numeric_cols], y)
    selected_features = numeric_cols[selector.get_support()]

    # Balanceamento
    if len(y.unique()) > 1 and y.value_counts()[0] / y.value_counts()[1] > 1.5:
        X_res, y_res = SMOTE(random_state=42).fit_resample(X_selected, y)
        print(f"Balanceamento aplicado. Nova distribuição: {pd.Series(y_res).value_counts().to_dict()}")
        return X_res, y_res, scaler, selector, selected_features

    return X_selected, y, scaler, selector, selected_features


def train_and_evaluate(X, y, selector, feature_names):
    """Treina e avalia o modelo KNN"""
    print("\n=== Modelagem KNN ===")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Otimização de hiperparâmetros
    param_grid = {
        'n_neighbors': np.arange(3, 30, 2),
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }

    grid = GridSearchCV(
        KNeighborsClassifier(),
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    grid.fit(X_train, y_train)

    print(f"\nMelhores parâmetros: {grid.best_params_}")
    print(f"Melhor acurácia (validação): {grid.best_score_:.2f}")

    # Avaliação final
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)

    print("\nRelatório de Classificação:")
    print(classification_report(y_test, y_pred))

    # Matriz de confusão
    plt.figure(figsize=(6, 6))
    sns.heatmap(
        confusion_matrix(y_test, y_pred),
        annot=True, fmt='d', cmap='Blues',
        xticklabels=['Ruim', 'Bom'],
        yticklabels=['Ruim', 'Bom']
    )
    plt.title('Matriz de Confusão')
    plt.show()

    # Validação cruzada
    cv_scores = cross_val_score(best_model, X, y, cv=10)
    print(f"\nAcurácia média (CV): {cv_scores.mean():.2f} (±{cv_scores.std():.2f})")

    # Importância das features (se disponível)
    if selector and hasattr(selector, 'scores_'):
        plt.figure(figsize=(10, 6))
        scores_selected = selector.scores_[selector.get_support()]  # Filtra apenas as selecionadas
        pd.Series(scores_selected, index=feature_names).sort_values().plot.barh()
        plt.title('Importância das Features (Selecionadas)')
        plt.show()

    return best_model


def main():
    """Fluxo principal de execução"""
    df = load_data()
    exploratory_analysis(df)
    df = feature_engineering(df)
    X, y, scaler, selector, feature_names = preprocess_data(df)
    model = train_and_evaluate(X, y, selector, feature_names)

    # Salvar artefatos
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/knn/knn_model.pkl')
    joblib.dump(scaler, 'models/knn/scaler.pkl')
    if selector:
        joblib.dump(selector, 'models/knn/selector.pkl')
    print("\nModelo e pré-processadores salvos na pasta 'models'")


if __name__ == "__main__":
    main()