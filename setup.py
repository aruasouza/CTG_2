from azure.datalake.store import core, lib, multithread
import pandas as pd
from datetime import datetime

tenant = '6e2475ac-18e8-4a6c-9ce5-20cace3064fc'
RESOURCE = 'https://datalake.azure.net/'
client_id = "0ed95623-a6d8-473e-86a7-a01009d77232"
client_secret = "NC~8Q~K~SRFfrd4yf9Ynk_YAaLwtxJST1k9S4b~O"
adlsAccountName = 'deepenctg'

adlCreds = lib.auth(tenant_id = tenant,
                client_secret = client_secret,
                client_id = client_id,
                resource = RESOURCE)

adlsFileSystemClient = core.AzureDLFileSystem(adlCreds, store_name=adlsAccountName)

today = datetime.now()
logfile_name = f'log_{today.month}_{today.year}.csv'

try:
    multithread.ADLDownloader(adlsFileSystemClient, lpath=logfile_name, 
        rpath=f'DataLakeRiscoECompliance/LOG/{logfile_name}', nthreads=64, 
        overwrite=True, buffersize=4194304, blocksize=4194304)

except FileNotFoundError:
    pd.DataFrame({'time':[],'output':[],'error':[]}).to_csv(logfile_name,index = False)
    multithread.ADLUploader(adlsFileSystemClient, lpath=logfile_name,
        rpath=f'DataLakeRiscoECompliance/LOG/{logfile_name}', nthreads=64, overwrite=True, buffersize=4194304, blocksize=4194304)

try:
    multithread.ADLDownloader(adlsFileSystemClient, lpath='records.csv', 
        rpath=f'DataLakeRiscoECompliance/LOG/records.csv', nthreads=64, 
        overwrite=True, buffersize=4194304, blocksize=4194304)

except FileNotFoundError:
    pd.DataFrame({'file_name':[],'origin':[]}).to_csv('records.csv',index = False)
    multithread.ADLUploader(adlsFileSystemClient, lpath='records.csv',
        rpath=f'DataLakeRiscoECompliance/LOG/records.csv', nthreads=64, overwrite=True, buffersize=4194304, blocksize=4194304)

def upload_file_to_directory(local_path,directory,file_name):
    multithread.ADLUploader(adlsFileSystemClient, lpath=local_path,
        rpath=f'{directory}/{file_name}', nthreads=64, overwrite=True, buffersize=4194304, blocksize=4194304)