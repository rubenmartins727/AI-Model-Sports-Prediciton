import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar o modelo treinado
modelo = joblib.load('modelo_treinado.pkl')

# Importância das características
importances = modelo.feature_importances_
feature_names = pd.read_csv('features.csv').columns

# Criar um DataFrame para as importâncias das características
feature_importances = pd.DataFrame({'feature': feature_names, 'importance': importances})
feature_importances = feature_importances.sort_values(by='importance', ascending=False)

# Plotar as 10 características mais importantes
plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importances.head(10))
plt.title('Top 10 Características mais Importantes')
plt.show()
