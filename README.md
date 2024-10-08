# Parking App Project

## Requirements

* [Python](https://www.python.org/).
* [Docker](https://www.docker.com/).
* [Poetry](https://python-poetry.org/) for Python package and environment management.

___
## Setup Project

* Make virtual environment: 

By default, the dependencies are managed with [Poetry](https://python-poetry.org/), go there and install it.

You can install all the dependencies with:

```console
$ poetry install
```
* Activate virtual environment:

Then you can start a shell session with the new environment with:

```console
$ poetry shell
```

Make sure your editor is using the correct Python virtual environment.

* Create your own `.env` file based on the template `.env.example`

___
## Running Application

* Start the stack with Docker Compose:

```bash
docker compose up -d
```
* Initialize alembic migrations:

```bash
alembic upgrade head
```
* Starting application:

```bash
py main.py
```

___
## Acessing on local
The application will get started in http://127.0.0.1:8000  

Swagger Documentation: http://127.0.0.1:8000/docs

Redoc Documentation: http://127.0.0.1:8000/redoc
___

