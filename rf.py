from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from utils import *

RANDOM_STATE = 42
TEST_SIZE = 0.2
OUTPUT_PATH = 'output/rf/'
WEIGHT_PATH = 'weights/rf/'

def train_and_evaluate(X, y):
    """Treina e avalia o modelo com GridSearch"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # Grid de parâmetros otimizado
    param_grid = {
        'n_estimators': [50, 100],       # Número de árvores (menos = mais rápido)
        'max_depth': [None, 10],         # Profundidade máxima
        'max_features': ['sqrt', 0.5]    # Features por árvore
    }

    grid = GridSearchCV(
        RandomForestClassifier(random_state=RANDOM_STATE),
        param_grid,
        cv=5,
        n_jobs=-1,
        scoring='accuracy'
    )
    grid.fit(X_train, y_train)

    # Melhor modelo
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)

    # Salvar métricas e modelo
    save_metrics(y_test, y_pred, grid, output_path=OUTPUT_PATH, name="rf")
    save_artifacts(best_model, path=WEIGHT_PATH)
    return best_model

if __name__ == "__main__":
    df = load_data()
    X, y = preprocess_data(df, "RF")
    model = train_and_evaluate(X, y)
