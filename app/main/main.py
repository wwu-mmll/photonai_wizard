import os
from flask import Flask
from flask_ldap3_login import LDAP3LoginManager
from flask_login import LoginManager
from flask_uploads import UploadSet, configure_uploads
from flask_wtf.csrf import CSRFProtect
from pymodm import connect

from .wizard_config import *


base_dir = os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
# base_dir = os.path.join(base_dir, 'photon-wizard')
base_dir = os.path.join(base_dir, 'app')
base_dir = os.path.join(base_dir, 'main')
template_dir = os.path.join(base_dir, 'templates')
static_dir = os.path.join(base_dir, 'static')
application = Flask(__name__, template_folder=template_dir, static_folder=static_dir)

application.config.from_object(PHOTONWebConfig)
# application.config.from_object(WizardTRAPConfig)
# application.config.from_object(DebugWizardConfig)
application.config.update(SESSION_COOKIE_NAME='wizard_cookie')
# prepare data csv file upload
# csrf = CSRFProtect(application)
excelfiles = UploadSet('excelfile', extensions=('xlsx'))
configure_uploads(application, excelfiles)

login_manager = LoginManager(application)              # Setup a Flask-Login Manager
login_manager.login_view = 'index'

# Setup a LDAP3 Login Manager.
if not application.config['LOGIN_DISABLED']:
    ldap_manager = LDAP3LoginManager(application)

connect(application.config["MONGODB_CS"]+"/photon-wizard2", alias="photon_wizard", connect=False)  #

# this line is important (add all controllers)
if application.config["UDO_MODE"]:
  from .controller import general, new_pipeline, login, ldap
else:
   from .controller import general, new_pipeline, login

