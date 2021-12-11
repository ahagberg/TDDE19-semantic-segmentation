FROM python:3.6-buster
COPY ./requirements.txt /app/requirements.txt
WORKDIR /app
RUN pip install -r requirements.txt
CMD tail -f /dev/null
