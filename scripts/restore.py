import os
import subprocess

from dotenv import load_dotenv

load_dotenv()

MONGO_DB = str(os.getenv('MONGO_DB'))
MONGO_DB_USERNAME = str(os.getenv('MONGO_DB_USERNAME'))
MONGO_DB_PASSWORD = str(os.getenv('MONGO_DB_PASSWORD'))

MONGO_URI = f'db uri in which you want to restore the backup'

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKUP_DIR = os.path.join(BASE_DIR, 'backups')
ARCHIVE = f'{os.path.join(BACKUP_DIR,MONGO_DB)}.gzip'

def restore():
    restore_command = f'mongorestore --uri="{MONGO_URI}" --archive="{ARCHIVE}" --gzip'

    process = subprocess.run(restore_command, shell=True, cwd='backup_binaries')

    if process.returncode == 0:
        print('restore successful')
    else:
        print('restore failed')



if __name__ == '__main__':
    restore()