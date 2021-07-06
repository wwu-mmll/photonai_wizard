from ..main import ldap_manager
from .login import load_user, QCUser


@ldap_manager.save_user
def save_user(dn, username, data, memberships):
    user = load_user(dn)
    if not user:
        user = QCUser(dn, username, data)
        user.save()
    return user

