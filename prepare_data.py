# prepare_data.py

import pandas as pd
import os
from sklearn.preprocessing import StandardScaler

def ler_arquivos_excel(diretorio):
    dataframes = []
    for subdir, _, files in os.walk(diretorio):
        for file in files:
            if file.endswith('.xlsx') or file.endswith('.xls'):
                caminho_arquivo = os.path.join(subdir, file)
                df = pd.read_excel(caminho_arquivo)
                dataframes.append(df)
    dados_completos = pd.concat(dataframes, ignore_index=True)
    return dados_completos

def preparar_dados(diretorio):
    dados_completos = ler_arquivos_excel(diretorio)

    if 'FTR' not in dados_completos.columns:
        raise KeyError("A coluna 'FTR' não está presente no DataFrame.")
    
    dados_completos = pd.get_dummies(dados_completos, columns=['FTR'], prefix='FTR')

    numeric_columns = dados_completos.select_dtypes(include=['float64', 'int64']).columns
    categorical_columns = dados_completos.select_dtypes(include=['object', 'category']).columns

    dados_completos[numeric_columns] = dados_completos[numeric_columns].fillna(dados_completos[numeric_columns].mean())

    for col in categorical_columns:
        dados_completos[col] = dados_completos[col].fillna(dados_completos[col].mode()[0])

    dados_completos['Date'] = pd.to_datetime(dados_completos['Date'], format='%d/%m/%Y')
    for col in categorical_columns:
        if col != 'Date':
            dados_completos[col] = dados_completos[col].astype('category')

    dados_completos = pd.concat([dados_completos, 
                                 dados_completos['Date'].dt.year.rename('Year'), 
                                 dados_completos['Date'].dt.month.rename('Month'), 
                                 dados_completos['Date'].dt.dayofweek.rename('DayOfWeek'), 
                                 (dados_completos['HS'] - dados_completos['AS']).rename('ShotDifference')], axis=1)

    scaler = StandardScaler()
    dados_completos[numeric_columns] = scaler.fit_transform(dados_completos[numeric_columns])

    dados_completos = pd.get_dummies(dados_completos, columns=categorical_columns, drop_first=True)

    colunas_para_remover = ['Date', 'Time', 'HomeTeam', 'AwayTeam', 'Div', 'Referee']
    colunas_existentes = [col for col in colunas_para_remover if col in dados_completos.columns]
    dados_completos = dados_completos.drop(columns=colunas_existentes)

    target_columns = ['FTR_H', 'FTR_D', 'FTR_A']
    X = dados_completos.drop(columns=target_columns)
    y = dados_completos['FTR_H']

    return X, y

if __name__ == "__main__":
    diretorio = 'C:/Users/rrmartins/OneDrive - myPartner - Consultoria Informática, S.A/Desktop/AI Model'
    X, y = preparar_dados(diretorio)
    X.to_csv('features.csv', index=False)
    y.to_csv('target.csv', index=False)
