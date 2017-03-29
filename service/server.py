# We'll render HTML templates and access data sent by POST
# using the request object from flask. Redirect and url_for
# will be used to redirect the user once the upload is done
# and send_from_directory will help us to send/show on the
# browser the file that the user just uploaded
from flask import Flask, request, url_for, render_template, send_from_directory, g
import flask
import sqlite3
import time

from werkzeug import secure_filename
import json
import os
import uuid
import cv2
import numpy as np

# Initialize the Flask application
app = Flask(__name__, template_folder='./templates/')

execfile('config.py')


def connect_db():
    return sqlite3.connect(DATABASE)


@app.before_request
def before_request():
    g.db = connect_db()
    g.db.row_factory = sqlite3.Row


@app.teardown_appcontext
def close_connection(exception):
    if hasattr(g, 'db'):
        g.db.close()


def query_db(query, args=(), one=False):
    cur = g.db.execute(query, args)
    rv = cur.fetchall()
    cur.close()
    return (rv[0] if rv else None) if one else rv


def init_db():
    with app.app_context():
        db = get_db()
        with app.open_resource('schema.sql', mode='r') as f:
            db.cursor().executescript(f.read())
        db.commit()


def insert(table, **kwargs):
    # g.db is the database connection
    cur = g.db.cursor()
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
    g.db.commit()
    id = cur.lastrowid

    cur.close()
    return id


# This route will show a form to perform an AJAX request
# jQuery is loaded to execute the request and update the
# value of the operation
@app.route('/')
def index():
    return render_template('index.html')


# Route that will process the file upload
@app.route('/upload', methods=['POST'])
def upload():
    # Get the name of the uploaded file
    file = request.files['file']
    # Make the filename safe, remove unsupported chars
    filename = secure_filename(file.filename)
    uniqueName = uuid.uuid4().hex
    fullPath = os.path.join(app.config['UPLOAD_FOLDER'], uniqueName)
    file.save(fullPath)

    try:
        capture = cv2.VideoCapture(fullPath)
        if capture.isOpened():
            insert('requests', uuid=uniqueName, status='INSERTED', original_name=filename)
            response = {}
            response['status'] = 'OK'
            response['uuid'] = uniqueName
            return flask.jsonify(**response)
    except:
        os.remove(fullPath)
        response = {}
        response['status'] = 'FAILED'
        response['message'] = 'Unable to open the provided file as video.'
        return flask.jsonify(**response), 415


@app.route('/status/<uuid>')
def status(uuid):
    response = {}
    response['status'] = 'FAILED'
    statusCode = 500

    result = query_db("SELECT * from requests where uuid=?", args=[uuid], one=True)
    if not result:
        response['message'] = 'Wrong video identifier {}'.format(uuid)
        statusCode = 404
    else:
        response['data'] = dict(result)
        response['data']['lastchange'] = response['data']['lastchange'] - time.time()
        response['data']['created'] = response['data']['created'] - time.time()

        jobStatus = list(query_db( "SELECT * from jobs where uuid=? and status='DONE'", args=[uuid]))
        response['data']['finished_stages'] = len(jobStatus)
        response['data']['stages'] = []
        jobStatus = query_db( "SELECT * from jobs where uuid=?", args=[uuid])
        for counter, stage in enumerate(jobStatus):
            stageInfo = {}
            stageInfo['id'] = counter
            stageInfo['stage'] = stage['identifier']
            stageInfo['status'] = stage['status']
            stageInfo['lastchange'] = stage['lastchange'] - time.time()
            stageInfo['start'] = stage['created'] - time.time()
            response['data']['stages'].append(stageInfo)
        response['status'] = 'OK'
        statusCode = 200

    return flask.jsonify(**response), statusCode


# This route is expecting a parameter containing the name
# of a file. Then it will locate that file on the upload
# directory and show it on the browser, so if the user uploads
# an image, that image is going to be show after the upload
@app.route('/video/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename + '.avi.avi')


def getFps(uuid):
    try:
        fullPath = os.path.join(app.config['UPLOAD_FOLDER'], uuid)
        capture = cv2.VideoCapture(fullPath)
        return capture.get(5)
    except:
        return 0


def getAggreagate(data, intervals, fps):
    for frame in data:
        frameTime = int(frame) / fps
        usedInts = [i for i in intervals if frameTime in i[0]]

        if usedInts:
            for personID in data[frame]:
                person = data[frame][personID]
                if 'emotions' in person:
                    values = np.asarray(person['emotions'])
                    for i in usedInts:
                        i[1] += 1
                        i[2] += values
    return intervals


def getOriginalName(uuidString):
    result = query_db(
        "SELECT original_name from requests where uuid=?",
        args=[uuidString], one=True)
    if not result:
        return None
    return result['original_name']


def createIntervalResponse(name, time, data):
    identifierlString = '{}#time={},{}'.format(name, time[0], time[1])
    data = data[2] / (data[1] + 0.001)
    valence = data[3] - (data[0] + data[1] + data[2] + data[4]) / 4.0
    arousal = 0.3 - data[6]
    result = {'anger': data[0],
              'disgust': data[1],
              'fear': data[2],
              'happiness': data[3],
              'sadness': data[4],
              'surprise': data[5]}
    if data[6] == 0:
        valence = 0
        arousal = 0

    valenceArousal = {
        '@type': 'emotionSet',
        'prov:wasGeneratedBy': 'videoAnalysis_vad',
        'onyx:hasEmotion': {
            '@type': 'emotion',
            'pad:arousal': arousal,
            'pad:pleasure': valence
        },
    }

    big6 = {
        '@type': 'emotionSet',
        'prov:wasGeneratedBy': 'videoAnalysis_category',
        'onyx:hasEmotion': [],
    }
    for emotion in result:
        if result[emotion] > 0.1:
            big6['onyx:hasEmotion'].append({
                '@type': 'emotion',
                'onyx:hasEmotionCategory': 'big6:{}'.format(emotion),
                'onyx:hasEmotionIntensity': result[emotion]
            })

    frame = {
        '@id': identifierlString,
        'emotions': [big6, valenceArousal]
    }
    return frame


def createAggResponse(uuidString, intervals):
    response = {}
    response['status'] = 'FAILED'
    statusCode = 500

    original_name = getOriginalName(uuidString)
    if not original_name:
        response['message'] = 'Unknown video identifier "{}"'.format(
            uuidString)
        statusCode = 404
        return response, statusCode
    fps = getFps(uuidString)
    try:
        fullPath = os.path.join(
            app.config['UPLOAD_FOLDER'], uuidString + '.json')
        print(fullPath)
        with open(fullPath, 'r') as f:
            data = json.load(f)
    except:
        response['message'] = 'Results not ready for {}'.format(uuidString)
        statusCode = 404
        return response, statusCode

    intervals = getAggreagate(data, intervals, fps)

    frame = {
        '@context': 'http://senpy.cluster.gsi.dit.upm.es/api/contexts/Results.jsonld',  
        # 'http://pchradis.fit.vutbr.cz:9000/results_agg',
        '@type': 'results',
        'analysis': ['videoAnalysis_category', 'videoAnalysis_vad'],
        'entries': []
    }

    for i in intervals:
        frame['entries'].append(createIntervalResponse(original_name, i[0][0], i))

    statusCode = 200

    return frame, statusCode


@app.route('/results_agg/<uuidString>', defaults={'timing': '0,inf'})
@app.route('/results_agg/<uuidString>/<timing>')
def results_agg(uuidString, timing):
    try:
        from interval import interval
        intervals = []
        for intervalStr in timing.split(';'):
            i = interval([float(x) for x in intervalStr.split(',')])
            intervals.append([i, 0, np.zeros(7)])
    except:
        response = {}
        response['status'] = 'FAILED'
        response['message'] = 'Unable to parse time intervals "{}"'.format(
            timing)
        statusCode = 404
        return flask.jsonify(**response), statusCode

    response, statusCode = createAggResponse(uuidString, intervals)
    return flask.jsonify(**response), statusCode


@app.route('/results_agg2/<uuid>', defaults={'timing': '0,inf'})
@app.route('/results_agg2/<uuid>/<timing>')
def results_agg2(uuid, timing):
    response = {}
    response['status'] = 'FAILED'
    statusCode = 500

    fps = getFps(uuid)
    if fps <= 0:
        response['message'] = 'Wrong video identifier {}'.format(uuid)
        statusCode = 404
        return flask.jsonify(**response), statusCode
    try:
        fullPath = os.path.join(app.config['UPLOAD_FOLDER'], uuid + '.json')
        with open(fullPath, 'r') as f:
            data = json.load(f)
    except:
        response['message'] = 'Results not ready for {}'.format(uuid)
        statusCode = 404
        return flask.jsonify(**response), statusCode

    from interval import interval

    intervals = []
    for intervalStr in timing.split(';'):
        i = interval([float(x) for x in intervalStr.split(',')])
        intervals.append([i, 0, np.zeros(7)])

    intervals = getAggreagate(data, intervals, fps)

    results = []
    for i in intervals:
        i[2] /= i[1]
        valence = i[2][3] - (i[2][0] + i[2][1] + i[2][2] + i[2][4]) / 4.0
        arousal = 0.3 - i[2][6]
        result = {'interval': '{},{}'.format(*i[0][0]),
                  'anger': '{:.3f}'.format(i[2][0]),
                  'disgust': '{:.3f}'.format(i[2][1]),
                  'fear': '{:.3f}'.format(i[2][2]),
                  'smile': '{:.3f}'.format(i[2][3]),
                  'sad': '{:.3f}'.format(i[2][4]),
                  'surprised': '{:.3f}'.format(i[2][5]),
                  'valence': '{:.3f}'.format(valence),
                  'arousal': '{:.3f}'.format(arousal)}
        results.append(result)
    response['data'] = results
    response['status'] = 'OK'
    statusCode = 200
    return flask.jsonify(**response), statusCode


@app.route('/results/<filename>')
def json_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename + '.json')


if __name__ == '__main__':
    app.run(
        host="0.0.0.0",
        port=int("9000"),
        debug=False,
        # threaded=True,
        processes=16,
        # use_reloader=True
    )
