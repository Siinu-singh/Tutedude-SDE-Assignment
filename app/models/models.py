from datetime import datetime
import json
from flask_sqlalchemy import SQLAlchemy

# Create db instance that will be initialized by app factory
db = SQLAlchemy()

class Candidate(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationship
    interviews = db.relationship('Interview', backref='candidate', lazy=True)

class Interview(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    candidate_id = db.Column(db.Integer, db.ForeignKey('candidate.id'), nullable=False)
    interviewer_name = db.Column(db.String(100), nullable=False)
    start_time = db.Column(db.DateTime, default=datetime.utcnow)
    end_time = db.Column(db.DateTime)
    duration = db.Column(db.Integer)  # in seconds
    video_path = db.Column(db.String(200))  # Path to recorded video file
    integrity_score = db.Column(db.Float, default=100.0)
    status = db.Column(db.String(20), default='active')  # active, completed, terminated
    
    # Relationship
    events = db.relationship('ProctorEvent', backref='interview', lazy=True)
    
    @property
    def duration_formatted(self):
        if self.duration:
            hours = self.duration // 3600
            minutes = (self.duration % 3600) // 60
            seconds = self.duration % 60
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        return "00:00:00"

class ProctorEvent(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    interview_id = db.Column(db.Integer, db.ForeignKey('interview.id'), nullable=False)
    event_type = db.Column(db.String(50), nullable=False)  # focus_loss, no_face, multiple_faces, phone_detected, notes_detected
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    duration = db.Column(db.Float)  # duration of the event in seconds
    confidence = db.Column(db.Float)  # confidence score for detection
    details = db.Column(db.Text)  # additional details in JSON format
    
    def get_details(self):
        if self.details:
            return json.loads(self.details)
        return {}
    
    def set_details(self, details_dict):
        self.details = json.dumps(details_dict)

class SystemLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    level = db.Column(db.String(20), nullable=False)  # INFO, WARNING, ERROR
    message = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    interview_id = db.Column(db.Integer, db.ForeignKey('interview.id'))