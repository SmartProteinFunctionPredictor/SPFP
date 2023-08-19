FROM python
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE ${env.process.PORT}
CMD gunicorn workers=4 --bind 0.0.0.0:${env.process.PORT} app:app
