# FROM tensorflow/tensorflow:1.15.0-py3
FROM python:3.6-jessie

RUN apt update
RUN apt-get update
RUN apt-get install -y libsm6 libxext6 libxrender-dev

WORKDIR /app
ADD requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

# RUN pip uninstall tensorboard -y
# RUN pip install tensorboard==1.15

ADD frozen_inference_graph.pb /app/frozen_inference_graph.pb

ADD . /app
ENV PORT 8080
CMD ["gunicorn", "app:app", "--config=config.py"]