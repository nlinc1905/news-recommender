# news-recommender
A Django website for browsing the latest news and recommended news based on user click history.

Its purpose is to serve as a minimal website to explore and test recommender systems.  

New users can be created with Django Admin, and each user will see different recommendations based on their click 
history.

The website is set up to run on 1 recommender model, as defined by the environment variable in docker-compose.

# How to Use It

To build and start the service for the first time:
```
docker-compose up --build
```

Go to http://localhost:8000

The recommender system is hosted as a separate API.  After starting the service, its Swagger docs can be found at: 
http://localhost:5000

## Admin/Super-User Credentials

A super-user is created when the service builds.  The credentials are in the docker-compose environment variables:
```
DJANGO_SUPERUSER_PASSWORD=adminPW123
DJANGO_SUPERUSER_USERNAME=admin
DJANGO_SUPERUSER_EMAIL=admin@admin.com
```

## Adding a New Recommender Model

The service is set up to work with models from Microsoft's recommenders Python library.  To add your own, you will need 
to modify recommender_system/src/models.py, add a folder in models/, and update the docker-compose environment variable.
