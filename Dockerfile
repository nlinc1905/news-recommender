# Pull base image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV NBR_TOP_STORIES 10

# Set work directory in docker container
WORKDIR /

# Install dependencies
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /
RUN pip install --upgrade pip setuptools && pip install -r requirements.txt

# Copy project to working dir in container
COPY . /

# Load fixtures to DB
RUN python3 startup_scripts/create_db_fixtures_for_mind_data.py
