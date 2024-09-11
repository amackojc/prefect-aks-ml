FROM tensorflow/tensorflow:2.17.0-gpu

ENV WORKDIR=/opt/prefect/prefect-aks/workflows

COPY data "$WORKDIR/data"
COPY requirements.txt "$WORKDIR/requirements.txt"

WORKDIR $WORKDIR

RUN python3 -m venv "$WORKDIR/venv"
ENV PATH="$WORKDIR/venv/bin:$PATH"

RUN pip install --no-cache-dir -r requirements.txt

LABEL io.prefect.version=3.0.0
