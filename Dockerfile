# Dockerfile, Image, Container
FROM python:3.9.13

ADD main.py models.py montecarlo.py setup.py .

RUN pip install ttkthemes fredapi python-bcb darts pandas numpy azure-datalake-store azure-mgmt-resource azure-mgmt-datalake-store

CMD [ "python", "./main.py" ]