# evaluate_model.py

import pandas as pd
import joblib
from sklearn.metrics import classification_report, accuracy_score

def avaliar_modelo():
    modelo = joblib.load('modelo_treinado.pkl')
    X_teste = pd.read_csv('features_teste.csv')
    y_teste = pd.read_csv('target_teste.csv').values.ravel()  # Corrigir o formato do y

    y_pred = modelo.predict(X_teste)
    
    print("Accuracy:", accuracy_score(y_teste, y_pred))
    print("Classification Report:\n", classification_report(y_teste, y_pred))

if __name__ == "__main__":
    avaliar_modelo()
