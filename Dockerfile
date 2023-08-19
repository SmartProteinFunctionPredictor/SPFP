FROM python
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE $3000
CMD gunicorn --bind 0.0.0.0:$3000 app:app
