#!/usr/bin/env python3
"""
Demo data generator for the Video Interview Proctoring System
Creates sample interview data for demonstration purposes
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app, db
from app.models import Candidate, Interview, ProctorEvent
from datetime import datetime, timedelta
import random

def create_demo_data():
    """Create sample interview data for demonstration"""
    
    app = create_app()
    
    with app.app_context():
        # Clear existing data
        db.drop_all()
        db.create_all()
        
        # Create sample candidates
        candidates = [
            Candidate(name="John Smith", email="john.smith@email.com"),
            Candidate(name="Sarah Johnson", email="sarah.johnson@email.com"),
            Candidate(name="Mike Chen", email="mike.chen@email.com"),
            Candidate(name="Emily Davis", email="emily.davis@email.com"),
            Candidate(name="Alex Rodriguez", email="alex.rodriguez@email.com")
        ]
        
        for candidate in candidates:
            db.session.add(candidate)
        
        db.session.commit()
        
        # Create sample interviews
        interviewers = ["Dr. Smith", "Prof. Johnson", "Ms. Wilson", "Mr. Brown", "Dr. Lee"]
        
        for i, candidate in enumerate(candidates):
            # Create completed interview
            start_time = datetime.now() - timedelta(days=random.randint(1, 30))
            end_time = start_time + timedelta(minutes=random.randint(30, 90))
            
            interview = Interview(
                candidate_id=candidate.id,
                interviewer_name=random.choice(interviewers),
                start_time=start_time,
                end_time=end_time,
                duration=int((end_time - start_time).total_seconds()),
                status='completed'
            )
            
            db.session.add(interview)
            db.session.commit()
            
            # Create sample events for the interview
            event_types = ['focus_loss', 'no_face', 'multiple_faces', 'phone_detected', 'notes_detected', 'drowsiness_detected']
            
            # Generate random events
            num_events = random.randint(0, 15)
            for j in range(num_events):
                event_time = start_time + timedelta(seconds=random.randint(0, interview.duration))
                event_type = random.choice(event_types)
                
                event = ProctorEvent(
                    interview_id=interview.id,
                    event_type=event_type,
                    timestamp=event_time,
                    duration=random.uniform(1, 10) if event_type in ['focus_loss', 'no_face', 'drowsiness_detected'] else 0,
                    confidence=random.uniform(0.6, 0.95)
                )
                
                # Set details based on event type
                if event_type == 'multiple_faces':
                    event.set_details({'face_count': random.randint(2, 4)})
                elif event_type in ['phone_detected', 'notes_detected']:
                    event.set_details({
                        'object_class': 'cell phone' if event_type == 'phone_detected' else 'book',
                        'bbox': [random.randint(50, 200), random.randint(50, 200), 
                                random.randint(50, 150), random.randint(50, 150)]
                    })
                else:
                    event.set_details({'threshold_exceeded': True})
                
                db.session.add(event)
            
            # Calculate integrity score
            events = ProctorEvent.query.filter_by(interview_id=interview.id).all()
            score = 100
            
            for event in events:
                if event.event_type == 'focus_loss':
                    score -= 2
                elif event.event_type == 'no_face':
                    score -= 5
                elif event.event_type == 'multiple_faces':
                    score -= 10
                elif event.event_type == 'phone_detected':
                    score -= 15
                elif event.event_type == 'notes_detected':
                    score -= 10
                elif event.event_type == 'drowsiness_detected':
                    score -= 8
            
            interview.integrity_score = max(0, min(100, score))
            
        db.session.commit()
        
        print("✅ Demo data created successfully!")
        print(f"📊 Created {len(candidates)} candidates with sample interviews")
        print("🎯 You can now access the dashboard to see sample data")

if __name__ == '__main__':
    create_demo_data()