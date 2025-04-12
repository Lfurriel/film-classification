from sklearn.neighbors import KNeighborsClassifier

from utils import *
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

RANDOM_STATE = 42
TEST_SIZE = 0.2
OUTPUT_PATH = 'output/knn/'
WEIGHT_PATH = 'weights/knn/'

def train_and_evaluate(X, y):
    """Treina e avalia o modelo KNN"""
    print("\n=== Modelagem KNN ===")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Otimização de hiperparâmetros
    param_grid = {
        'n_neighbors': np.arange(3, 39, 2),
        'weights': ['uniform'], # ['uniform', 'distance']
        'metric': ['euclidean'] # ['euclidean', 'manhattan']
    }

    grid = GridSearchCV(
        KNeighborsClassifier(),
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    grid.fit(X_train, y_train)

    cv_scores = cross_val_score(grid.best_estimator_, X_train, y_train, cv=10)
    print(f"\nAcurácia média (CV): {cv_scores.mean():.2f} (±{cv_scores.std():.2f})")

    print(f"\nMelhores parâmetros: {grid.best_params_}")
    print(f"Melhor acurácia (validação): {grid.best_score_:.2f}")

    # Avaliação final
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)

    save_metrics(y_test, y_pred, grid, output_path=OUTPUT_PATH, name="knn")
    save_artifacts(best_model, WEIGHT_PATH)

    return best_model

if __name__ == "__main__":
    df = load_data()
    X, y = preprocess_data(df, "KNN")
    model = train_and_evaluate(X, y)