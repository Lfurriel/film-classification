from utils import *
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV

RANDOM_STATE = 42
TEST_SIZE = 0.2
OUTPUT_PATH = 'output/xgb/'
WEIGHT_PATH = 'weights/xgb/'

def train_and_evaluate(X, y, X_processed):
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=42, stratify=y
    )

    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0],
        'gamma': [0, 0.1, 0.2],
        'colsample_bytree': [0.8, 1.0]
    }

    xgb_base = XGBClassifier(eval_metric='logloss')  # Parâmetro corrigido

    grid = GridSearchCV(
        estimator=xgb_base,
        param_grid=param_grid,
        scoring='accuracy',
        cv=5,
        verbose=1,
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)

    save_metrics(y_test, y_pred, grid, output_path=OUTPUT_PATH)
    save_artifacts(best_model, path=WEIGHT_PATH)
    return best_model

if __name__ == "__main__":
    df = load_data()
    print("Tamanho total do dataset:", len(df))
    X, y, X_processed = preprocess_data(df, "XGB")
    model = train_and_evaluate(X, y, X_processed)