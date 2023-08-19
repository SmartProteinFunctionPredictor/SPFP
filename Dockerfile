FROM python:3.11
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE $PORT
CMD gunicorn --bind 0.0.0.0:$PORT --timeout 300 app:app
