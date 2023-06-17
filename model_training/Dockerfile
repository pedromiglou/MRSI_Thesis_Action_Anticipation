FROM tensorflow/tensorflow:latest-gpu

WORKDIR /model_training

COPY ./requirements.txt /model_training/requirements.txt

RUN pip install -r /model_training/requirements.txt

COPY utils.py /model_training/utils.py
COPY points_conv_1d.py /model_training/model_training.py

CMD ["python3", "model_training.py"]