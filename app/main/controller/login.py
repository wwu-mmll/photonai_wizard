from flask import render_template, redirect, url_for, session
from ..main import application, login_manager
from ..model.elements import QCUser
from flask_ldap3_login.forms import LDAPLoginForm
from flask_login import login_user, logout_user, current_user
from pymodm.errors import DoesNotExist

import os


@application.route('/login', methods=['GET', 'POST'])
def login():
    # Instantiate a LDAPLoginForm which has a validator to check if the user
    # exists in LDAP.
    form = LDAPLoginForm()

    # if request.method == "POST":
    # login_user(User(dn='uid=rleenings,ou=users,dc=uni-muenster,dc=de', username='rleenings', data={}))
    if form.validate_on_submit():
        # Successfully logged in, We can now access the saved user object
        # via form.user.
        login_user(form.user)  # Tell flask-login to log them in.
        session['user'] = current_user.username

        # # the user has no tutorial copies yet, copy tutorial pipelines to user's repo
        # try:
        #     pipe = Pipeline.objects.raw({'name': 'Breast Cancer', 'user': current_user.username}).first()
        # except DoesNotExist as e:
        #     define_tutorials(current_user.username)
        # ---> obsolete, we copy the pipeline now everytime the user clicks on the tutorial button

        if not os.path.isdir("/spm-data/Scratch/photon_wizard/" + session['user']):
            os.mkdir("/spm-data/Scratch/photon_wizard/" + session['user'])

        return redirect(url_for('project_history'))  # Send them home

    return render_template('index.html', form=form)



# Declare a User Loader for Flask-Login.
# Simply returns the User if it exists in our 'database', otherwise
# returns None.
@login_manager.user_loader
def load_user(id):
    try:
        user = QCUser.objects.raw({'dn': id}).first()
        return user
    except DoesNotExist:
        return None


@application.route("/logout")
def logout():
    logout_user()
    return redirect(url_for('index'))
