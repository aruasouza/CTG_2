from azure.datalake.store import core, lib, multithread

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

file_2 = 'Projecoes_LCA.xlsx'

multithread.ADLDownloader(adlsFileSystemClient, lpath=file_2, 
        rpath=f'DataLakeRiscoECompliance/ETL/{file_2}', nthreads=64, 
        overwrite=True, buffersize=4194304, blocksize=4194304)