FROM python:3.6-jessie
RUN apt update
WORKDIR /app
ADD requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

RUN pip uninstall tensorboard -y
RUN pip install tensorboard==1.15

RUN gsutil cp gs://happy-walrus-api.appspot.com/frozen_inference_graph.pb frozen_inference_graph.pb
ADD frozen_inference_graph.pb

ADD . /app
ENV PORT 8080
CMD ["gunicorn", "app:app", "--config=config.py"]