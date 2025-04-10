import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
import joblib

# Configuração para exibir mais colunas no pandas
pd.set_option('display.max_columns', None)

# Carregar o dataset
df = pd.read_csv('files/dataset_preprocessado.csv', encoding='UTF-8')
df = df.drop(columns=['vote_average'])

# Verificar as primeiras linhas do dataset
print("\nPrimeiras linhas do dataset:")
print(df.head())

# Verificar informações básicas
print("\nInformações do dataset:")
print(df.info())

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

# Dividir em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


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


# 1. Árvore de Decisão com critério Gini
print("\n=== Árvore de Decisão (Gini) ===")
dt_gini = DecisionTreeClassifier(criterion='gini', random_state=42)
dt_gini.fit(X_train, y_train)
evaluate_model(dt_gini, X_test, y_test)

# Visualizar a árvore (limitada a profundidade 3 para melhor visualização)
plt.figure(figsize=(20, 10))
plot_tree(dt_gini, max_depth=3, feature_names=features, class_names=['Ruim', 'Bom'], filled=True)
plt.title("Árvore de Decisão (Gini) - Primeiros 3 níveis")
plt.show()

# 2. Árvore de Decisão com critério Entropia
print("\n=== Árvore de Decisão (Entropia) ===")
dt_entropy = DecisionTreeClassifier(criterion='entropy', random_state=42)
dt_entropy.fit(X_train, y_train)
evaluate_model(dt_entropy, X_test, y_test)

# Visualizar a árvore (limitada a profundidade 3 para melhor visualização)
plt.figure(figsize=(20, 10))
plot_tree(dt_entropy, max_depth=3, feature_names=features, class_names=['Ruim', 'Bom'], filled=True)
plt.title("Árvore de Decisão (Entropia) - Primeiros 3 níveis")
plt.show()

# 3. Encontrar os melhores hiperparâmetros para a Árvore de Decisão
print("\n=== Otimização de Hiperparâmetros para Árvore de Decisão ===")
param_grid = {
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy']
}

dt = DecisionTreeClassifier(random_state=42)
grid_search = GridSearchCV(dt, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

print(f"Melhores parâmetros: {grid_search.best_params_}")
print(f"Melhor score na validação cruzada: {grid_search.best_score_:.2f}")

# Treinar com os melhores parâmetros encontrados
best_dt = grid_search.best_estimator_
print("\n=== Melhor Árvore de Decisão ===")
evaluate_model(best_dt, X_test, y_test)

# 4. Random Forest
print("\n=== Random Forest ===")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
evaluate_model(rf, X_test, y_test)

# Encontrar o melhor número de árvores para a Random Forest
print("\n=== Encontrando o melhor número de árvores para Random Forest ===")
vscore = []
vn = []
for n in range(10, 210, 20):
    model = RandomForestClassifier(n_estimators=n, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    print(f'Número de árvores: {n} - Acurácia: {score:.2f}')
    vscore.append(score)
    vn.append(n)

best_n = vn[np.argmax(vscore)]
print(f'\nMelhor número de árvores: {best_n} com acurácia: {max(vscore):.2f}')

plt.figure(figsize=(10, 5))
plt.plot(vn, vscore, '-bo')
plt.xlabel('Número de Árvores')
plt.ylabel('Acurácia')
plt.title('Desempenho da Random Forest por Número de Árvores')
plt.show()

# 5. Importância das Features
print("\n=== Importância das Features ===")
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(12, 8))
plt.title("Importância das Features")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), np.array(features)[indices], rotation=90)
plt.xlabel("Features")
plt.ylabel("Importância")
plt.tight_layout()
plt.show()

# Mostrar as features mais importantes
print("\nFeatures mais importantes:")
for f in range(X.shape[1]):
    print(f"{f + 1}. {features[indices[f]]}: {importances[indices[f]]:.4f}")

joblib.dump(best_dt, 'models/rf_model.pkl')
print("\nModelo salvo!")