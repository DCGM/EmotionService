import sqlite3
import time
import os
import sys
from utils import query_db
from subprocess import call

execfile('config.py')

conn = sqlite3.connect(DATABASE)
conn.row_factory = sqlite3.Row

def runScript( command):
    print os.getcwd()
    sys.stdout.flush()
    sys.stdout.flush()
    print command.split()
    command = command.split()
    status = call('pwd')
    status = call(command)
    print "Finished", command, status
    return status == 0

def main():
    print "START"
    sys.stdout.flush()

    while True:
        job = query_db(conn,
            "SELECT * FROM jobs  WHERE status='PENDING' ORDER BY lastchange ASC",
             one=True)
        if job:
            c = conn.cursor()
            c.execute(
                "UPDATE jobs SET status='RUNNING', lastchange=? WHERE uuid=? and status='PENDING'",
                (time.time(), job['uuid']))
            conn.commit()

            if c.rowcount:
                result = runScript(job['command'])
                status = 'FINISHED' if result else 'FAILED'
                c.execute(
                    "UPDATE jobs SET status=?, lastchange=? WHERE id=?",
                    (status, time.time(), job['id']))
                conn.commit()

        time.sleep(0.2)


if __name__ == '__main__':
    main()
