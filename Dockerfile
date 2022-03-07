FROM python:3.6

# update ca-certificates
RUN apt-get update -y && apt-get install ca-certificates -y


ADD requirements.txt requirements.txt
ADD app /app
RUN chmod 777 /app/main/static/tmp/

RUN pip install --upgrade pip
RUN pip install --no-cache-dir --upgrade -r requirements.txt
RUN pip install gunicorn

# register photonai_neuro since non privileged user cannot write to PHOTONAI registry
RUN python -c "import photonai_neuro"

RUN useradd --create-home flaskserver
USER flaskserver
 
ENV STATIC_URL /main/static
# Absolute path in where the static files wil be
ENV STATIC_PATH /app/main/static

WORKDIR /
ENV PYTHONPATH = /

EXPOSE 8080

CMD ["gunicorn", "--conf", "/app/gunicorn_conf.py", "--bind", "0.0.0.0:8080", "app.main"]
