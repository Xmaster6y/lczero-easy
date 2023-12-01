# syntax=docker/dockerfile:1
FROM python:3.9.18

WORKDIR /code

RUN apt-get update && apt-get install -y \
    ocl-icd-opencl-dev \
    libopenblas-dev \
    zip

RUN pip install --upgrade pip
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    python-multipart

RUN mkdir -p /service/api
COPY api /service/api
COPY bin/lc0 /bin/
EXPOSE 8000
COPY docker/start.sh ./
RUN chmod +x start.sh
ENTRYPOINT ["/code/start.sh"]
