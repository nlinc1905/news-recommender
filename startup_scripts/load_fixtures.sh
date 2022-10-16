#!/bin/bash

python3 manage.py collectstatic --noinput

python3 manage.py makemigrations
python3 manage.py migrate

python3 manage.py createsuperuser --noinput

python3 manage.py loaddata users/fixtures/customuser
python3 manage.py loaddata articles/fixtures/articles
python3 manage.py loaddata articles/fixtures/topstories
python3 manage.py loaddata articles/fixtures/userhistory

# gunicorn newsfinder.wsgi -b 0.0.0.0:8000
gunicorn --bind 0.0.0.0:8000 newsfinder.asgi -w 4 -k uvicorn.workers.UvicornWorker
