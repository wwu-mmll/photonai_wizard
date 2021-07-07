
class PHOTONWebConfig(object):
    UDO_MODE = False
    SECRET_KEY = 'do-i-really-need-this'
    # THE MONGODB THAT THE WIZARD ELEMENTS ARE STORED IN, NOT PHOTONAI RESULTS
    # specify your mongodb instance here
    MONGODB_CS = 'mongodb://localhost:27017'
    LOGIN_DISABLED = True

    # specify a folder where the PHOTONAI script can be written to
    # this is necessary to make it available as download afterwards
    TMP_PHOTON_SCRIPT_FOLDER = '/app/main/static/tmp'
    UPLOAD_FOLDER = '/app/main/uploads'
    UPLOADS_DEFAULT_DEST = 'app/main/uploads'
