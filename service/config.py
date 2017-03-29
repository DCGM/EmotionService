DATABASE = './database.db'
JOB_FILE = './job_configuration'

try:
    # This is the path to the upload directory
    app.config['UPLOAD_FOLDER'] = './uploads/'
    app.config['MAX_CONTENT_LENGTH'] = 60 * 1024 * 1024
except:
    pass
