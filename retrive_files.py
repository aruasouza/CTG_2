from azure.datalake.store import core, lib, multithread
import os

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

file_links = ["DataLakeRiscoECompliance/ETL/Projecoes_LCA.xlsx",
                "DataLakeRiscoECompliance/ETL/deepen_teste_web.xlsx",
                "DataLakeRiscoECompliance/ETL/projecao_de_investiamento/DCF - RP'22 8&04 - Valores.xlsx",
                "DataLakeRiscoECompliance/ETL/projecao_de_investiamento/Saldo de Caixa 8&4.xlsx",
                "DataLakeRiscoECompliance/ETL/bpc/Memória de Cálculo Inflação.xlsx",
                "DataLakeRiscoECompliance/ETL/calculo_de_divida/09 Debt vs Setembro 2022 - CTGBR - Budget 2023.xlsx",
                "DataLakeRiscoECompliance/ETL/calculo_de_divida/09 Debt vs Setembro 2022 - RP - Budget 2023.xlsx",
                "DataLakeRiscoECompliance/ETL/calculo_de_divida/Budget 2023 - INP_RESULT_FINANC_TESO_DEBT - Novo formulário - Valores.xlsx"]

for file in file_links:
        try:
                multithread.ADLDownloader(adlsFileSystemClient, lpath=os.path.join('datalake_files',file.split('/')[-1]), 
                        rpath=file, nthreads=64, 
                        overwrite=True, buffersize=4194304, blocksize=4194304)
        except Exception as e:
                print(e)