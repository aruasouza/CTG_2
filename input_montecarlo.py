from tkinter import filedialog
from tkinter import ttk
from azure.datalake.store import multithread
import pandas as pd
from datetime import datetime
import re
from setup import upload_file_to_directory,logfile_name,adlsFileSystemClient

def upload_file(Risco,root):
    file_path = filedialog.askopenfilename()
    try:
        output = pd.read_csv(file_path)
    except Exception:
        ttk.Label(root,text = 'Erro: O arquivo selecionado não é do formato correto.').place(relx=0.5, rely=0.2, anchor='center')
        return
    if len(output) > 72:
        ttk.Label(root,text = 'Erro: O tamanho do arquivo excede o limite máximo').place(relx=0.5, rely=0.2, anchor='center')
        return
    if list(output.columns) != ['date', 'prediction', 'superior', 'inferior', 'std']:
        ttk.Label(root,text = 'Erro: As colunas do arquivo devem ser (nessa ordem): date, prediction, superior, inferior, std').place(relx=0.5, rely=0.2, anchor='center')
        return
    date_sample = output.loc[0,'date'] 
    if not re.match("^\d{4}-\d{2}$", date_sample):
        ttk.Label(root,text = 'Erro: As datas não estão no formato correto (YYYY-mm). Exemplo: 2020-04').place(relx=0.5, rely=0.2, anchor='center')
        return
    time = datetime.now()
    time_str = str(time).replace('.','-').replace(':','-').replace(' ','-')
    file_name = f'{Risco}_{time_str}.csv'
    # Colocando o output para csv e encaminhando-o para o data lake
    upload_file_to_directory(file_path,f'DataLakeRiscoECompliance/PrevisionData/Variables/{Risco}/Manual',file_name)
    log = pd.read_csv(logfile_name)
    log = pd.concat([log,pd.DataFrame({'time':[time],'output':[file_name],'error':['no errors']})])
    log.to_csv(logfile_name,index = False)
    multithread.ADLUploader(adlsFileSystemClient, lpath=logfile_name,
        rpath=f'DataLakeRiscoECompliance/LOG/{logfile_name}', nthreads=64, overwrite=True, buffersize=4194304, blocksize=4194304)    
    records = pd.read_csv('records.csv')
    records = pd.concat([records,pd.DataFrame({'file_name':[file_name],'origin':['Manual']})])
    records.to_csv('records.csv',index = False)
    multithread.ADLUploader(adlsFileSystemClient, lpath='records.csv',
        rpath=f'DataLakeRiscoECompliance/LOG/records.csv', nthreads=64, overwrite=True, buffersize=4194304, blocksize=4194304)