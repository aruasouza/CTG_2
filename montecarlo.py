import numpy as np
import pandas as pd
import random
import os
from azure.datalake.store import multithread
from setup import adlsFileSystemClient

def distribution(media,std,minimo,maximo,n):
    dist = list(np.random.normal(media,std,n))
    for i,valor in enumerate(dist):
        if not (minimo <= valor <= maximo):
            dist[i] = (random.random() * (maximo - minimo)) + minimo
    return dist

def find_files(risco):
    records = pd.read_csv('records.csv')
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

def linhas_limite(data,simulation):
    pass

def simulate(mes,ano):
    data = f'{ano}-{mes}'
    linha = main_dataframe.loc[main_dataframe['date'] == data]
    media = linha['prediction'].iloc[0]
    maximo = linha['superior'].iloc[0]
    minimo = linha['inferior'].iloc[0]
    std = main_dataframe.loc[0,'std']
    global simulation
    simulation = distribution(media,std,minimo,maximo,1000)