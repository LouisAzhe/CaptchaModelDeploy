FROM python:3.7.12-buster
MAINTAINER AzheLouis "https://github.com/LouisAzhe"

WORKDIR /app
ADD app.py /app
ADD requirements.txt /app
ADD uvicorn_config.json /app
ADD densenet121_ep50_fulldata.pkl /app
ADD Captcha_Model.py /app

RUN pip install -r requirements.txt

EXPOSE 9003
ENV TZ "Asia/Taipei"
ENTRYPOINT ["python", "app.py"]
