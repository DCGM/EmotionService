import sqlite3
import time
from utils import query_db
from collections import OrderedDict

execfile('config.py')

conn = sqlite3.connect(DATABASE)
conn.row_factory = sqlite3.Row


def runScript():
    pass


def readJobs(filename):
    jobs = OrderedDict()
    last = None
    with open(filename, 'r') as f:
        for line in f:
            words = line.split()
            if len(words) and line[0] != '#':
                jobs[words[0]] = ' '.join(words[1:])
                if last:
                    jobs[last] = (jobs[last], words[0])
                last = words[0]
        jobs[last] = (jobs[last], 'DONE')
    return jobs


def insert(conn, table, **kwargs):
    cur = conn.cursor()
    fields = [key for key in kwargs]
    values = [kwargs[key] for key in kwargs]
    fields.append('lastchange')
    values.append(time.time())
    fields.append('created')
    values.append(time.time())
    query = 'INSERT INTO %s (%s) VALUES (%s)' % (
        table,
        ', '.join(fields),
        ', '.join(['?'] * len(values))
    )
    cur.execute(query, values)
    conn.commit()
    id = cur.lastrowid
    cur.close()
    return id


def insertJOB(conn, jobID, next_job, jobCommand, requestUUID):
    jobCommand = jobCommand.format(ID=requestUUID)
    insert(conn, table='jobs', uuid=requestUUID,
           command=jobCommand, status='PENDING',
           identifier=jobID, next_job=next_job)


def processNewRequests(jobs, conn):
    pendingRequests = query_db(
        conn, "SELECT * FROM requests  WHERE status='INSERTED'")
    for pendingRequest in pendingRequests:
        c = conn.cursor()
        c.execute(
            "UPDATE requests SET status='PROCESSING', lastchange=? WHERE uuid=? and status='INSERTED'",
            (time.time(), pendingRequest['uuid']))
        conn.commit()

        if c.rowcount:
            jobID = jobs.keys()[0]
            insertJOB(
                conn, jobID=jobID, next_job=jobs[jobID][1],
                jobCommand=jobs[jobID][0], requestUUID=pendingRequest['uuid'])


def processFinishedJobs(jobs, conn):
    finishedJobs = query_db(
        conn, "SELECT * FROM jobs  WHERE status='FINISHED'")
    for finishedJob in finishedJobs:
        print('HAVE')
        c = conn.cursor()
        c.execute(
            "UPDATE jobs SET status='DONE', lastchange=? WHERE id=? and status='FINISHED'",
            (time.time(), finishedJob['id']))
        conn.commit()
        if c.rowcount:
            nextJobID = finishedJob['next_job']
            if nextJobID != 'DONE':
                insertJOB(
                    conn, jobID=nextJobID, next_job=jobs[nextJobID][1],
                    jobCommand=jobs[nextJobID][0],
                    requestUUID=finishedJob['uuid'])
            else:
                c = conn.cursor()
                c.execute(
                    "UPDATE requests SET status='DONE', lastchange=? WHERE uuid=?",
                    (time.time(), finishedJob['uuid']))
                conn.commit()


def processFailedJobs(jobs, conn):
    finishedJobs = query_db(
        conn, "SELECT * FROM jobs  WHERE status='FAILED'")
    for finishedJob in finishedJobs:
        c = conn.cursor()
        c.execute(
            "UPDATE jobs SET status='ERROR', lastchange=? WHERE id=? and status='FAILED'",
            (time.time(), finishedJob['id']))
        conn.commit()
        if c.rowcount:
            nextJobID = finishedJob['next_job']
            c = conn.cursor()
            c.execute(
                "UPDATE requests SET status='ERROR', lastchange=? WHERE uuid=?",
                (time.time(), finishedJob['uuid']))
            conn.commit()


def main():
    jobs = readJobs(JOB_FILE)
    while True:
        processNewRequests(jobs, conn)
        processFinishedJobs(jobs, conn)
        processFailedJobs(jobs, conn)
        time.sleep(0.2)


if __name__ == '__main__':
    main()
