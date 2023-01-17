import numpy as np
import pandas as pd
import os
from azure.datalake.store import multithread
from setup import adlsFileSystemClient
from bcb import sgs,currency
from datetime import date

def absolute(serie):
    valor_atual = 1598.41
    yield valor_atual
    for valor in serie[1:]:
        valor = valor / 100
        valor_atual += (valor_atual * valor)
        yield valor_atual

def get_current_ipca():
    try:
        dataframe = sgs.get({'ipca_change':433},start = '2000-01-01')
    except:
        raise TimeoutError('Erro de conexão com o Banco Central')
    dataframe['indice'] = [valor for valor in absolute(dataframe['ipca_change'].values)]
    return dataframe['indice'].values[-1]

def get_current_cambio():
    try:
        cy = currency.get('USD', start = '2022-01-01',end = str(date.today()))
    except:
        raise TimeoutError('Erro de conexão com o Banco Central')
    return cy.values.ravel()[-1]

def get_current_selic():
    try:
        dataframe = sgs.get({'selic':432}, start = '2022-01-01')
    except:
        raise TimeoutError('Erro de conexão com o Banco Central')
    return dataframe.values.ravel()[-1]

def find_files(risco):
    records = pd.read_csv('records.csv')
    global contracts
    try:
        filtered_df = records.loc[[x.find(risco) != -1 for x in records['file_name']]].copy()
        file_names = filtered_df['file_name'].values[-10:]
        origins = filtered_df['origin'].values[-10:]
        full_strings = (filtered_df['file_name'].apply(lambda x: x[:-4]) + filtered_df['origin'].apply(lambda x: ', ' + x)).values[-10:]
        global main_info
        main_info = {'risco':risco,'size':len(full_strings),'file_names':file_names,'origins':origins,'full_strings':full_strings}
    except IndexError:
        print('Não existem simulações de inflação')

def find_datas(index):
    file_name = main_info['file_names'][index]
    origin = main_info['origins'][index]
    risco = main_info['risco']
    multithread.ADLDownloader(adlsFileSystemClient, lpath=file_name, 
        rpath=f'DataLakeRiscoECompliance/PrevisionData/Variables/{risco}/{origin}/{file_name}', nthreads=64, 
        overwrite=True, buffersize=4194304, blocksize=4194304)
    global main_dataframe
    main_dataframe = pd.read_csv(file_name)
    main_dataframe['ano'] = main_dataframe['date'].apply(lambda x: x.split('-')[0])
    main_dataframe['mes'] = main_dataframe['date'].apply(lambda x: x.split('-')[1])
    os.remove(file_name)

def impact():
    pass

def simulate(mes,ano):
    data_final = f'{ano}-{mes}'
    std = main_dataframe['std'].iloc[0]
    df_pred = main_dataframe[['date','prediction']].copy()
    df_pred = df_pred.set_index(pd.to_datetime(df_pred['date'],format = '%Y-%m'))
    global simulation
    simulation = np.random.normal(size = 10000)
    # contracts = pd.read_excel('Valores_cambio.xlsx')
    # contracts = contracts.set_index(pd.to_datetime(contracts['data'].apply(lambda x: x[3:]),format = '%m/%Y'))
    
