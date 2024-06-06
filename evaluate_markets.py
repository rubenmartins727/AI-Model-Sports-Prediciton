import pandas as pd
import numpy as np
import joblib

# Função para calcular a expectativa de retorno para cada mercado
def calcular_expectativa_retorno(dados, mercado, modelo, features_importantes):
    retornos = []
    for i, row in dados.iterrows():
        # Extrair as características importantes
        caracteristicas = row[features_importantes].values.reshape(1, -1)
        # Fazer a previsão usando o modelo treinado
        predicao = modelo.predict(caracteristicas)[0]
        resultado_real = row['resultado_real']
        odds = row[mercado]

        # Calcular o retorno da aposta
        if predicao == resultado_real:
            retorno = (odds - 1) * 10  # Supondo uma aposta de 10 unidades
        else:
            retorno = -10
        
        retornos.append(retorno)
    
    # Calcular a expectativa de retorno
    expectativa_retorno = np.mean(retornos)
    return expectativa_retorno

def main():
    # Carregar os dados de teste com odds e previsões
    dados_completos = pd.read_csv('dados_completos_com_odds_teste.csv')

    # Características mais importantes usadas no treinamento do modelo refinado
    features_importantes = ['FTHG', 'FTAG', 'PSCA', 'HTHG', 'HTAG', 'HS', 'AS', 'HST', 'B365H', 'B365D', 'B365A', 'VCCH', 'AvgCH', 'HTR_H', 'MaxCH']

    # Verificar se as colunas necessárias estão presentes
    required_columns = ['resultado_real', 'B365H', 'B365D', 'B365A', 'PSCA', 'VCCH'] + features_importantes
    for col in required_columns:
        if col not in dados_completos.columns:
            raise ValueError(f"Coluna necessária '{col}' não encontrada no DataFrame")

    # Carregar o modelo refinado
    modelo_refinado = joblib.load('xgb_model_refinado.pkl')

    # Mercados de apostas a serem avaliados
    mercados = ['FTHG', 'FTAG', 'PSCA', 'HTHG', 'HTAG', 'HS', 'AS', 'HST', 'B365H', 'B365D', 'B365A', 'VCCH', 'AvgCH', 'HTR_H', 'MaxCH']

    # Calcular a expectativa de retorno para cada mercado
    expectativas_retorno = {mercado: calcular_expectativa_retorno(dados_completos, mercado, modelo_refinado, features_importantes) for mercado in mercados}

    # Imprimir os resultados
    print("Expectativa de Retorno para cada Mercado de Apostas:")
    for mercado, expectativa in expectativas_retorno.items():
        print(f"{mercado}: {expectativa:.2f}")

if __name__ == "__main__":
    main()
