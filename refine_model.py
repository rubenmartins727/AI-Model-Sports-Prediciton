import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier
import joblib

# Carregar os dados completos com odds
dados_completos = pd.read_csv('dados_completos_com_odds.csv')

# Verificar as colunas disponíveis e ajustar a lista de características importantes
print(dados_completos.columns)

# Atualizar a lista de características importantes com base nas colunas disponíveis
features_importantes = ['FTHG', 'FTAG', 'PSCA', 'HTHG', 'HTAG', 'HS', 'AS', 'HST', 'B365H', 'B365D', 'B365A', 'VCCH', 'AvgCH', 'HTR_H', 'MaxCH']

# Garantir que as colunas de odds também estão presentes
odds_columns = ['B365H', 'B365D', 'B365A', 'PSCA', 'VCCH', 'AvgCH', 'HTR_H', 'MaxCH']
for col in odds_columns:
    if col not in dados_completos.columns:
        raise ValueError(f"Coluna de odds necessária '{col}' não encontrada no DataFrame")

# Selecionar as características e o alvo
target_column = 'FTR_H'
X = dados_completos[features_importantes]
y = dados_completos[target_column]

# Dividir os dados em treino e teste
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=42)

# Definir a grade de parâmetros
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 1.0]
}

# Instanciar o modelo
xgb = XGBClassifier(random_state=42)

# Realizar Grid Search
grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_treino, y_treino)

# Salvar o melhor modelo
best_model = grid_search.best_estimator_
joblib.dump(best_model, 'xgb_model_refinado.pkl')

# Salvar os dados de teste com odds
dados_completos_teste = X_teste.copy()
dados_completos_teste['resultado_real'] = y_teste
dados_completos_teste.to_csv('dados_completos_com_odds_teste.csv', index=False)

print("Melhores hiperparâmetros:", grid_search.best_params_)

