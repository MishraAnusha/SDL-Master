# SDL Project

## Quick Start

To get this project up and running locally on your computer follow the following steps.

1. Clone Project Repository
2. go to project folder
3. Set up a python virtual environment
4. create .env file in project directory
5. Run the following commands
    - `pip install -r requirements.txt`
    - `python manage.py makemigrations`
    - `python manage.py migrate`
    - `python manage.py createsuperuser`
    - `python manage.py runserver 127.0.0.1:8000`
6. Open a browser and go to `http://127.0.0.1:8000/`

![project_structure](https://user-images.githubusercontent.com/25881570/172018369-7c8167d6-cf4e-487b-8e9d-3a1fc51d4e63.png)

### commands

- create new virtual environment
    - python -m venv env
- activate/deactivate environment
    - env\Scripts\Activate
    - source my_env/bin/activate
    - deactivate
- txt file for version
    - pip freeze > requirements.txt
- start project
    - python manage.py startapp name

### mongo db

https://www.mongodb.com/docs/manual/tutorial/install-mongodb-on-ubuntu/

- python manage.py makemigrations
- python manage.py migrate
- start mongo server : sudo systemctl start mongod
- mongo
- use iot_db

### CronJob

https://gutsytechster.wordpress.com/2019/06/24/how-to-setup-a-cron-job-in-django/

- add cron jobs : python manage.py crontab add
- active cron job list : python manage.py crontab show
- remove cron job : python manage.py crontab remove

### Help tutorials

- https://dev.to/earthcomfy/creating-a-django-registration-login-app-part-i-1di5

### Env key

- SECRET_KEY=Ur/J74Z5JAnLl70SKwKIqIUpH4CTnq+nppW+o8SNBjw=
- DEBUG=True
- MONGO_DB=db_name
- MONGO_DB_USERNAME=db_username
- MONGO_DB_PASSWORD=db_password
- EMAIL_USE_TLS=True
- EMAIL_PORT=587
- EMAIL_HOST='smtp.zoho.in'
- EMAIL_USER=email_username
- EMAIL_PASSWORD=password