import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib

# Carregar modelo e scaler
knn = joblib.load('models/knn_model.pkl')
scaler = joblib.load('models/scaler.pkl')

# Carregar dados
df = pd.read_csv('files/dataset_preprocessado.csv')

# Remover as mesmas colunas do treinamento
cols_to_drop = [
    'bom_ruim', 'original_title', 'release_date', 'original_language',
    'prod_company_1', 'prod_company_2', 'prod_company_3',
    'prod_country_1', 'prod_country_2', 'prod_country_3'
]

X = df.drop(cols_to_drop, axis=1, errors='ignore')
y = df['bom_ruim']

# Aplicar scaler
X_scaled = scaler.transform(X)

# Prever e avaliar
predictions = knn.predict(X_scaled)
print(f"\nAcurácia: {accuracy_score(y, predictions):.2f}")
print("Matriz de Confusão:")
print(confusion_matrix(y, predictions))