FROM tensorflow/tensorflow:1.15.0-py3
# FROM tensorflow/tensorflow-gpu:1.15.0-py3
# FROM python:3.6-jessie

RUN apt update
RUN apt-get update
RUN apt-get install -y libsm6 libxext6 libxrender-dev

# # Install CUDA
# RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_10.0.130-1_amd64.deb
# RUN dpkg -i cuda-repo-ubuntu1604_10.0.130-1_amd64.deb
# RUN apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
# RUN apt-get update
# RUN apt-get install cuda

WORKDIR /app
ADD requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

# RUN pip uninstall tensorboard -y
# RUN pip install tensorboard==1.15

ADD frozen_inference_graph.pb /app/frozen_inference_graph.pb

ADD . /app
ENV PORT 8080
CMD ["gunicorn", "app:app", "--config=config.py"]