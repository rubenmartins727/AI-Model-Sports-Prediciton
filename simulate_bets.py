import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Função para simular uma estratégia de apostas
def simular_apostas(dados, estrategia):
    saldo = 1000  # Saldo inicial
    aposta = 10  # Valor de cada aposta
    resultados = []

    for i, row in dados.iterrows():
        predicao = row['predicao']
        resultado_real = row['resultado_real']
        odds_favorito = row['odds_favorito']
        odds_nao_favorito = row['odds_nao_favorito']
        
        if estrategia == 'favorito':
            if predicao == 1:
                if resultado_real == 1:
                    saldo += aposta * (odds_favorito - 1)
                else:
                    saldo -= aposta
        elif estrategia == 'nao_favorito':
            if predicao == 0:
                if resultado_real == 0:
                    saldo += aposta * (odds_nao_favorito - 1)
                else:
                    saldo -= aposta
        
        resultados.append(saldo)
    
    return resultados

# Função para calcular ROI
def calcular_roi(saldo_inicial, saldo_final, num_apostas):
    investimento_total = num_apostas * 10  # Supondo 10 unidades por aposta
    retorno_total = saldo_final - saldo_inicial
    roi = (retorno_total / investimento_total) * 100
    return roi

def main():
    # Carregar o modelo treinado
    modelo = joblib.load('modelo_treinado.pkl')

    # Carregar os dados de teste
    X_teste = pd.read_csv('features_teste.csv')
    y_teste = pd.read_csv('target_teste.csv').values.ravel()

    # Fazer previsões no conjunto de teste
    dados_completos = X_teste.copy()
    dados_completos['predicao'] = modelo.predict(X_teste)
    dados_completos['resultado_real'] = y_teste
    
    # Usar odds reais do Bet365
    # Supondo que as colunas de odds no seu DataFrame são: B365H (odds para vitória em casa), B365A (odds para vitória fora)
    dados_completos['odds_favorito'] = dados_completos.apply(lambda row: row['B365H'] if row['predicao'] == 1 else row['B365A'], axis=1)
    dados_completos['odds_nao_favorito'] = dados_completos.apply(lambda row: row['B365A'] if row['predicao'] == 1 else row['B365H'], axis=1)
    
    # Simular estratégia de apostar no favorito
    resultados_favorito = simular_apostas(dados_completos, 'favorito')

    # Simular estratégia de apostar no não-favorito
    resultados_nao_favorito = simular_apostas(dados_completos, 'nao_favorito')

    # Plotar os resultados
    plt.plot(resultados_favorito, label='Favorito')
    plt.plot(resultados_nao_favorito, label='Não Favorito')
    plt.xlabel('Número de Apostas')
    plt.ylabel('Saldo')
    plt.title('Simulação de Estratégias de Apostas')
    plt.legend()
    plt.show()

    # Calcular ROI para as estratégias simuladas
    roi_favorito = calcular_roi(1000, resultados_favorito[-1], len(resultados_favorito))
    roi_nao_favorito = calcular_roi(1000, resultados_nao_favorito[-1], len(resultados_nao_favorito))

    print(f'ROI Favorito: {roi_favorito:.2f}%')
    print(f'ROI Não Favorito: {roi_nao_favorito:.2f}%')

if __name__ == "__main__":
    main()
