#!/bin/bash

aim init --repo /aim
aimlflow sync --mlflow-tracking-uri="http://mlflow.mlflow.svc.cluster.local:5000" --aim-repo=/aim &
sleep 15
aim server --repo /aim &
sleep 15
aim up --host 0.0.0.0 --repo /aim
