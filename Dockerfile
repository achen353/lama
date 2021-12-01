FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

ENTRYPOINT []

COPY ./bin/predict.py /work/bin/predict.py
COPY ./configs/prediction/default.yaml /work/configs/prediction/default.yaml
COPY ./models /work/models
COPY ./saicinpainting /work/saicinpainting
COPY ./scripts /work/scripts
COPY ./requirements.txt /work/requirements.txt

WORKDIR /work
RUN chmod +x ./scripts/install.sh
RUN ./scripts/install.sh

WORKDIR /work