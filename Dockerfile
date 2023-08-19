FROM python
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE $process.env.PORT
CMD gunicorn --bind 0.0.0.0:$process.env.PORT app:app
