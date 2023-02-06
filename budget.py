from setup import adlsFileSystemClient,multithread
import pandas as pd
import os
import datetime
import montecarlo
from dateutil.relativedelta import relativedelta
import numpy as np

file_links = ["DataLakeRiscoECompliance/ETL/projecao_de_investiamento/DCF - RP'22 8&04 - Valores.xlsx"]

for file in file_links:
    try:
        multithread.ADLDownloader(adlsFileSystemClient, lpath=os.path.join('datalake_files',file.split('/')[-1]), 
                rpath=file, nthreads=64, 
                overwrite=True, buffersize=4194304, blocksize=4194304)
    except Exception as e:
        print(e)

df = pd.read_excel("datalake_files/DCF - RP'22 8&04 - Valores.xlsx",sheet_name = 'Budget',header = 5)
df = df.loc[~df['Unnamed: 1'].isnull()].drop('Unnamed: 0',axis = 1).set_index('Unnamed: 1').T
df.columns.name = None
df = df[list(map(lambda x: type(x) == datetime.datetime,df.index.values))]
serie = df['Cash Accumulated Available']

def simulate(risco):
    serie_risco = montecarlo.main_dataframe['prediction'][:montecarlo.date_selected]
    print(serie_risco)
    first_date = serie_risco.index[0]
    limited_serie = serie[first_date:montecarlo.date_selected + relativedelta(months = 1)]
    print(limited_serie)
    std = montecarlo.main_dataframe['std'].iloc[0]
    cenarios_risco = [serie_risco + (std * value) for value in montecarlo.simulation]
    pior = '?'
    melhor = '?'
    if (risco == 'JUROS') and (len(limited_serie) == len(serie_risco)):
        cenarios = np.array([sum(limited_serie.values * (cenario.values / 100)) for cenario in cenarios_risco])
        pior = int(np.percentile(cenarios,25))
        melhor = int(np.percentile(cenarios,75))
    return {'cenarios':cenarios_risco,'pior':pior,'melhor':melhor}