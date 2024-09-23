FROM python:3.12

RUN pip install --no-cache-dir aim
RUN pip install --no-cache-dir aim-mlflow
RUN pip install --no-cache-dir azureml-mlflow

COPY startup.sh /aim/startup.sh
WORKDIR /aim

ENTRYPOINT ["/bin/sh", "-c"]
RUN chmod +x /aim/startup.sh

CMD ["/aim/startup.sh"]
