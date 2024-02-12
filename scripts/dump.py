import os
import subprocess

from dotenv import load_dotenv

load_dotenv()


MONGO_DB = str(os.getenv('MONGO_DB'))
MONGO_DB_USERNAME = str(os.getenv('MONGO_DB_USERNAME'))
MONGO_DB_PASSWORD = str(os.getenv('MONGO_DB_PASSWORD'))

MONGO_URI = f'mongodb+srv://{MONGO_DB_USERNAME}:{MONGO_DB_PASSWORD}@iot-db.ldpqm.mongodb.net/'

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKUP_DIR = os.path.join(BASE_DIR, 'backups')
ARCHIVE = f'{os.path.join(BACKUP_DIR,MONGO_DB)}.gzip'

def backup():
    # create backup directory if it doesn't exist
    if not os.path.exists(BACKUP_DIR):
        os.makedirs(BACKUP_DIR)

    dump_command = f'mongodump --uri="{MONGO_URI}" --db="{MONGO_DB}" --archive="{ARCHIVE}" --gzip'

    process = subprocess.run(dump_command, shell=True, cwd='backup_binaries')

    if process.returncode == 0:
        print('Backup successful')
    else:
        print('Backup failed')


if __name__ == '__main__':
    backup()