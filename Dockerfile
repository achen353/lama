FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

ENTRYPOINT []

COPY ./src/bin/predict.py /work/src/bin/predict.py
COPY ./src/configs/prediction/default.yaml /work/src/configs/prediction/default.yaml
COPY ./src/models /work/src/models
COPY ./src/saicinpainting /work/src/saicinpainting
COPY ./src/app /work/src/app
COPY ./src/lama.py /work/src/lama.py
COPY ./src/main.py /work/src/main.py
COPY ./scripts /work/scripts
COPY ./requirements.txt /work/requirements.txt

WORKDIR /work
RUN chmod +x ./scripts/install.sh
RUN ./scripts/install.sh

WORKDIR /work/src