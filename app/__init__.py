"""
Video Interview Proctoring System
Main application package
"""

from flask import Flask
from flask_session import Session
from app.core.config import Config

# Initialize session
session = Session()

def create_app():
    """Application factory pattern"""
    app = Flask(__name__)
    app.config.from_object(Config)
    
    # Import and initialize db from models
    from app.models.models import db
    db.init_app(app)
    
    # Initialize session
    session.init_app(app)
    
    # Register blueprints
    from app.api.routes import main_bp
    app.register_blueprint(main_bp)
    
    return app

# Import db for external use
from app.models.models import db