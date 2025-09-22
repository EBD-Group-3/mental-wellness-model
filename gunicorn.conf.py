# Gunicorn configuration file for Mental Wellness API
# For production deployment with optimal performance settings

import multiprocessing
import os

# Server socket
bind = "0.0.0.0:8000"
backlog = 2048

# Worker processes
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000
max_requests = 1000
max_requests_jitter = 50
preload_app = True
timeout = 120
keepalive = 2

# Restart workers after this many requests, with up to 50 requests variation
# This helps prevent memory leaks
max_requests = 1000
max_requests_jitter = 50

# Restart workers after this many seconds
max_worker_lifetime = 3600
max_worker_lifetime_jitter = 300

# Logging
accesslog = "-"  # Log to stdout
errorlog = "-"   # Log to stderr
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Process naming
proc_name = "mental-wellness-api"

# Security
limit_request_line = 4096
limit_request_fields = 100
limit_request_field_size = 8190

# SSL (uncomment for HTTPS)
# keyfile = "/path/to/keyfile"
# certfile = "/path/to/certfile"

# Environment variables for configuration
user = os.getenv('GUNICORN_USER', None)
group = os.getenv('GUNICORN_GROUP', None)
tmp_upload_dir = os.getenv('GUNICORN_TMP_UPLOAD_DIR', None)

# Preload application for better memory usage
preload_app = True

# Enable automatic worker restarts on code changes (development only)
reload = os.getenv('GUNICORN_RELOAD', 'false').lower() == 'true'

def when_ready(server):
    """Called just after the server is started."""
    server.log.info("Mental Wellness API server is ready. Listening on: %s", server.address)

def worker_int(worker):
    """Called just after a worker has been killed."""
    worker.log.info("Mental Wellness API worker received INT or QUIT signal")

def on_exit(server):
    """Called just before the server is shut down."""
    server.log.info("Mental Wellness API server is shutting down")

def on_starting(server):
    """Called just before the master process is initialized."""
    server.log.info("Mental Wellness API server is starting")

def on_reload(server):
    """Called to recycle workers during a reload via SIGHUP."""
    server.log.info("Mental Wellness API server is reloading")

def worker_abort(worker):
    """Called when a worker received the SIGABRT signal."""
    worker.log.info("Mental Wellness API worker received SIGABRT signal")