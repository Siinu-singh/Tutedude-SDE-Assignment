"""Utility functions for the proctoring system"""

from flask import current_app
from app import db
from app.models import ProctorEvent

def get_event_message(event):
    """Get human-readable message for an event"""
    messages = {
        'focus_loss': 'Candidate looking away from screen',
        'no_face': 'No face detected in frame',
        'multiple_faces': 'Multiple faces detected',
        'phone_detected': 'Mobile phone detected',
        'notes_detected': 'Notes or books detected',
        'drowsiness_detected': 'Drowsiness or eye closure detected',
        'audio_violation': 'Background voices detected'
    }
    return messages.get(event['type'], 'Unknown event')

def calculate_integrity_score(interview_id):
    """Calculate integrity score based on events"""
    events = ProctorEvent.query.filter_by(interview_id=interview_id).all()
    
    score = current_app.config['BASE_SCORE']
    
    for event in events:
        if event.event_type == 'focus_loss':
            score -= current_app.config['FOCUS_LOSS_PENALTY']
        elif event.event_type == 'no_face':
            score -= current_app.config['NO_FACE_PENALTY']
        elif event.event_type == 'multiple_faces':
            score -= current_app.config['MULTIPLE_FACES_PENALTY']
        elif event.event_type == 'phone_detected':
            score -= current_app.config['PHONE_DETECTION_PENALTY']
        elif event.event_type == 'notes_detected':
            score -= current_app.config['NOTES_DETECTION_PENALTY']
        elif event.event_type == 'drowsiness_detected':
            score -= 8  # Drowsiness penalty
        elif event.event_type == 'audio_violation':
            score -= 5  # Background voice penalty
    
    return max(0, min(100, score))  # Ensure score is between 0 and 100

def create_tables():
    """Create database tables"""
    db.create_all()
    
    # Create upload directory
    import os
    os.makedirs(current_app.config['UPLOAD_FOLDER'], exist_ok=True)