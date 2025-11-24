# app_ext/__init__.py

from datetime import datetime

from config import Config
from .downloads import cleanup_downloads
from .jobs import cleanup_jobs
from .routes_media import bp as media_bp
from .routes_youtube import bp as youtube_bp
from .routes_jobs import bp as jobs_bp

_LAST_CLEAN = datetime.utcnow()


def init_app(app):
    """Register blueprints and global hooks on the Flask app."""
    app.config.setdefault("SECRET_KEY", Config.SECRET_KEY)

    app.register_blueprint(media_bp)
    app.register_blueprint(youtube_bp)
    app.register_blueprint(jobs_bp)

    @app.before_request
    def periodic_cleanup():
        global _LAST_CLEAN
        now = datetime.utcnow()
        if (now - _LAST_CLEAN).total_seconds() >= Config.CLEANUP_INTERVAL_SEC:
            cleanup_downloads(now)
            cleanup_jobs(now)
            _LAST_CLEAN = now