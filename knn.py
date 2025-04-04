import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import joblib

# Configuração para exibir mais colunas no pandas
pd.set_option('display.max_columns', None)

# Carregar o dataset
df = pd.read_csv('files/dataset_preprocessado.csv', encoding='UTF-8')

# Verificar as primeiras linhas do dataset
print("\nPrimeiras linhas do dataset:")
print(df.head())

# Verificar distribuição da classe target
print("\nDistribuição da classe 'bom_ruim':")
print(df['bom_ruim'].value_counts())

# Pré-processamento dos dados
# Selecionar features - vamos usar todas as colunas numéricas exceto a target
features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
features.remove('bom_ruim')  # Remover a coluna target

# Separar features e target
X = df[features]
y = df['bom_ruim']

# Tratar valores NaN (substituir pela média da coluna)
X = X.fillna(X.mean())

# Normalizar os dados (importante para KNN)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


# Função para avaliar o modelo
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    # Acurácia
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAcurácia: {accuracy:.2f}")

    # Relatório de classificação
    print("\nRelatório de Classificação:")
    print(classification_report(y_test, y_pred))

    # Matriz de confusão
    print("Matriz de Confusão:")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues',
                xticklabels=['Ruim', 'Bom'],
                yticklabels=['Ruim', 'Bom'])
    plt.xlabel('Previsto')
    plt.ylabel('Real')
    plt.title('Matriz de Confusão')
    plt.show()


# 1. KNN com k=1 (como no exemplo)
print("\n=== KNN com k=1 ===")
knn1 = KNeighborsClassifier(n_neighbors=1)
knn1.fit(X_train, y_train)
evaluate_model(knn1, X_test, y_test)

# 2. KNN com k ótimo (usando validação cruzada)
print("\n=== Encontrando o melhor k ===")
param_grid = {'n_neighbors': np.arange(1, 30)}
knn = KNeighborsClassifier()
knn_cv = GridSearchCV(knn, param_grid, cv=5)
knn_cv.fit(X_train, y_train)

print(f"Melhor k encontrado: {knn_cv.best_params_['n_neighbors']}")
print(f"Melhor score na validação cruzada: {knn_cv.best_score_:.2f}")

# Treinar com o melhor k
best_k = knn_cv.best_params_['n_neighbors']
knn_best = KNeighborsClassifier(n_neighbors=best_k)
knn_best.fit(X_train, y_train)

print("\n=== KNN com melhor k ===")
evaluate_model(knn_best, X_test, y_test)

# 3. Validação cruzada completa
print("\n=== Validação Cruzada (5 folds) ===")
cv_scores = cross_val_score(knn_best, X_scaled, y, cv=5)
print(f"Scores de validação cruzada: {cv_scores}")
print(f"Média dos scores: {cv_scores.mean():.2f} (+/- {cv_scores.std() * 2:.2f})")

joblib.dump(knn_best, 'models/knn_model.pkl')
print("\nModelo salvo!")