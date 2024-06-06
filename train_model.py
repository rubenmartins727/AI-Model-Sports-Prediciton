# train_model.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

def treinar_modelo():
    X = pd.read_csv('features.csv')
    y = pd.read_csv('target.csv').values.ravel()  # Corrigir o formato do y

    X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=42)

    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(X_treino, y_treino)

    best_model = grid_search.best_estimator_
    joblib.dump(best_model, 'modelo_treinado.pkl')

    # Salvar os conjuntos de teste
    X_teste.to_csv('features_teste.csv', index=False)
    pd.DataFrame(y_teste, columns=['target']).to_csv('target_teste.csv', index=False)

    return best_model, X_teste, y_teste

if __name__ == "__main__":
    modelo, X_teste, y_teste = treinar_modelo()
    y_pred = modelo.predict(X_teste)
    
    print("\nMelhores hiperparâmetros encontrados:")
    print(modelo.get_params())
    print("\nDesempenho no conjunto de teste:")
    print("Accuracy:", accuracy_score(y_teste, y_pred))
    print("Classification Report:\n", classification_report(y_teste, y_pred))
