import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib

# Carregar dataset
df = pd.read_csv('files/dataset_preprocessado.csv')

# Remover colunas não numéricas e não relevantes
cols_to_drop = [
    'bom_ruim', 'original_title', 'release_date', 'original_language',
    'prod_company_1', 'prod_company_2', 'prod_company_3',
    'prod_country_1', 'prod_country_2', 'prod_country_3'
]

X = df.drop(cols_to_drop, axis=1, errors='ignore')
y = df['bom_ruim']

# Verificar tipos de dados
print("Tipos de dados restantes:")
print(X.dtypes)

# Dividir dados
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizar dados
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# Treinar modelo
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Salvar modelo e scaler
joblib.dump(knn, 'models/knn_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')

print("Modelo treinado e salvo com sucesso!")

print(classification_report(y_test, y_train))
#print("Matriz de Confusão:\n", confusion_matrix(y_test, y_pred))
acuracia = accuracy_score(y_test, y_train)
print(f"Acurácia do modelo: {acuracia:.2f}")

scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')
print(f"Acurácia média em cross-validation: {scores.mean():.2f} (± {scores.std():.2f})")

# Matriz de confusão
print("\nMatriz de Confusão:\n", confusion_matrix(y_test,y_train))