import os
from datetime import timedelta

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'tutedude-proctoring-system-2024'
    
    # SQLite Database (inbuilt)
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///proctoring.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Session Configuration
    SESSION_TYPE = 'filesystem'
    SESSION_PERMANENT = False
    PERMANENT_SESSION_LIFETIME = timedelta(hours=2)
    
    # Upload Configuration
    UPLOAD_FOLDER = 'static/recordings'
    MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500MB max file size
    
    # Proctoring Thresholds
    FOCUS_LOSS_THRESHOLD = 5  # seconds
    NO_FACE_THRESHOLD = 10    # seconds
    
    # Scoring Configuration
    BASE_SCORE = 100
    FOCUS_LOSS_PENALTY = 2
    NO_FACE_PENALTY = 5
    MULTIPLE_FACES_PENALTY = 10
    PHONE_DETECTION_PENALTY = 15
    NOTES_DETECTION_PENALTY = 10