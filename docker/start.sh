#!/bin/bash

echo "----------- Launching uvicorn --------- "
cd /service/

echo "----------- Run uvicon --------- "
uvicorn api.main:app --port 8000 --host 0.0.0.0
