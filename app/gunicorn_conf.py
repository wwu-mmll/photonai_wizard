# Gunicorn config variables
loglevel = "info"
errorlog = "-"  # stderr
#accesslog = "-"  # stdout
accesslog = None
worker_tmp_dir = "/dev/shm"
graceful_timeout = 120
timeout = 120
keepalive = 5
threads = 3