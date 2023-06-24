FROM tensorflow/tensorflow:latest-gpu

WORKDIR /model_training

COPY ./points /model_training/points

COPY ./requirements.txt /model_training/requirements.txt

RUN pip3 install -r /model_training/requirements.txt

COPY utils.py /model_training/utils.py
#COPY points_conv_1d.py /model_training/model_training.py
COPY points_conv_1d_testing.py /model_training/model_training.py
#COPY points_conv_2d.py /model_training/model_training.py
#COPY points_transformers.py /model_training/model_training.py


RUN mkdir /model_training/results

CMD ["python3", "/model_training/model_training.py"]