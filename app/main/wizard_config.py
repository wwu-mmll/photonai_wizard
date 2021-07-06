
class PHOTONWebConfig(object):
    UDO_MODE = False
    SECRET_KEY = 'do-i-really-need-this'
    # THE MONGODB THAT THE WIZARD ELEMENTS ARE STORED IN, NOT PHOTONAI RESULTS
    # specify your mongodb instance here
    MONGODB_CS = 'mongodb://SOME_MONGO_DB:27017'
    LOGIN_DISABLED = True

    # specify a folder where the PHOTONAI script can be written to
    # this is necessary to make it available as download afterwards
    TMP_PHOTON_SCRIPT_FOLDER = '/PATH/TO/LOCAL/FOLDER'
    UPLOAD_FOLDER = '/PATH/TO/LOCAL/FOLDER'
    UPLOADS_DEFAULT_DEST = '//PATH/TO/LOCAL/FOLDER'
