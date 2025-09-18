from flask import Blueprint, render_template, request, jsonify, redirect, url_for, session, send_file
import cv2
import base64
import numpy as np
import os
import json
from datetime import datetime, timedelta
from io import BytesIO

from app import db
from app.models.models import Candidate, Interview, ProctorEvent, SystemLog
from app.core.proctoring_engine import EnhancedProctoringEngine as ProctoringEngine
from app.utils.helpers import get_event_message, calculate_integrity_score

main_bp = Blueprint('main', __name__)

# Initialize proctoring engine
proctoring_engine = ProctoringEngine()

@main_bp.route('/')
def index():
    """Landing page"""
    return render_template('index_new.html')

@main_bp.route('/start_interview', methods=['GET', 'POST'])
def start_interview():
    """Start a new interview session"""
    if request.method == 'POST':
        candidate_name = request.form['candidate_name']
        candidate_email = request.form['candidate_email']
        interviewer_name = request.form['interviewer_name']
        
        # Check if candidate exists
        candidate = Candidate.query.filter_by(email=candidate_email).first()
        if not candidate:
            candidate = Candidate(name=candidate_name, email=candidate_email)
            db.session.add(candidate)
            db.session.commit()
        
        # Create new interview
        interview = Interview(
            candidate_id=candidate.id,
            interviewer_name=interviewer_name,
            start_time=datetime.utcnow()
        )
        db.session.add(interview)
        db.session.commit()
        
        # Store in session
        session['interview_id'] = interview.id
        session['candidate_name'] = candidate_name
        session['interviewer_name'] = interviewer_name
        
        return redirect(url_for('main.interview_room'))
    
    return render_template('start_interview.html')

@main_bp.route('/interview_room')
@main_bp.route('/interview_room/<int:interview_id>')
def interview_room(interview_id=None):
    """Main interview room with video proctoring"""
    # If interview_id provided in URL, use it
    if interview_id:
        interview = Interview.query.get(interview_id)
        if interview and interview.status == 'active':
            # Update session
            session['interview_id'] = interview_id
            session['candidate_name'] = interview.candidate.name
            session['interviewer_name'] = interview.interviewer_name
        else:
            return redirect(url_for('main.start_interview'))
    
    # Check session
    elif 'interview_id' not in session:
        # Try to find the most recent active interview
        recent_interview = Interview.query.filter_by(status='active').order_by(Interview.start_time.desc()).first()
        if recent_interview:
            session['interview_id'] = recent_interview.id
            session['candidate_name'] = recent_interview.candidate.name
            session['interviewer_name'] = recent_interview.interviewer_name
            interview = recent_interview
        else:
            return redirect(url_for('main.start_interview'))
    else:
        interview_id = session['interview_id']
        interview = Interview.query.get(interview_id)
        
        if not interview:
            # Clear invalid session and redirect
            session.pop('interview_id', None)
            return redirect(url_for('main.start_interview'))
    
    return render_template('interview_room.html', 
                         interview=interview,
                         candidate_name=session.get('candidate_name', interview.candidate.name),
                         interviewer_name=session.get('interviewer_name', interview.interviewer_name))

@main_bp.route('/process_frame', methods=['POST'])
def process_frame():
    """Process video frame for proctoring analysis"""
    try:
        if 'interview_id' not in session:
            return jsonify({'error': 'No active interview'}), 400
        
        interview_id = session['interview_id']
        
        # Get frame data
        frame_data = request.json['frame']
        
        # Decode base64 image
        header, encoded = frame_data.split(',', 1)
        frame_bytes = base64.b64decode(encoded)
        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({'error': 'Invalid frame data'}), 400
        
        # Process frame with proctoring engine
        analysis_result = proctoring_engine.process_frame(frame)
        
        # Store events in database
        for event in analysis_result['events']:
            proctor_event = ProctorEvent(
                interview_id=interview_id,
                event_type=event['type'],
                timestamp=event['timestamp'],
                duration=event.get('duration', 0),
                confidence=event.get('confidence', 1.0)
            )
            proctor_event.set_details(event.get('details', {}))
            db.session.add(proctor_event)
        
        if analysis_result['events']:
            db.session.commit()
        
        # Prepare response
        response_data = {
            'status': 'success',
            'events': [
                {
                    'type': event['type'],
                    'timestamp': event['timestamp'].isoformat(),
                    'message': get_event_message(event)
                }
                for event in analysis_result['events']
            ],
            'face_count': len(analysis_result['faces']),
            'objects_detected': len(analysis_result['objects'])
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Error processing frame: {e}")
        return jsonify({'error': str(e)}), 500

@main_bp.route('/end_interview', methods=['POST'])
def end_interview():
    """End the current interview session"""
    if 'interview_id' not in session:
        return jsonify({'error': 'No active interview'}), 400
    
    interview_id = session['interview_id']
    interview = Interview.query.get(interview_id)
    
    if interview:
        interview.end_time = datetime.utcnow()
        interview.duration = int((interview.end_time - interview.start_time).total_seconds())
        interview.status = 'completed'
        
        # Calculate integrity score
        interview.integrity_score = calculate_integrity_score(interview_id)
        
        db.session.commit()
        
        # Clear session
        session.pop('interview_id', None)
        
        return jsonify({
            'status': 'success',
            'interview_id': interview_id,
            'integrity_score': interview.integrity_score
        })
    
    return jsonify({'error': 'Interview not found'}), 404

@main_bp.route('/report/<int:interview_id>')
def generate_report(interview_id):
    """Generate proctoring report for an interview"""
    interview = Interview.query.get_or_404(interview_id)
    events = ProctorEvent.query.filter_by(interview_id=interview_id).all()
    
    # Categorize events
    event_summary = {
        'focus_loss': [],
        'no_face': [],
        'multiple_faces': [],
        'phone_detected': [],
        'notes_detected': [],
        'drowsiness_detected': [],
        'audio_violation': []
    }
    
    for event in events:
        if event.event_type in event_summary:
            event_summary[event.event_type].append(event)
    
    # Calculate statistics
    stats = {
        'total_focus_loss_events': len(event_summary['focus_loss']),
        'total_no_face_events': len(event_summary['no_face']),
        'total_multiple_faces_events': len(event_summary['multiple_faces']),
        'total_phone_detections': len(event_summary['phone_detected']),
        'total_notes_detections': len(event_summary['notes_detected']),
        'total_drowsiness_events': len(event_summary['drowsiness_detected']),
        'total_audio_violations': len(event_summary['audio_violation']),
        'total_focus_loss_time': sum([e.duration or 0 for e in event_summary['focus_loss']]),
        'total_no_face_time': sum([e.duration or 0 for e in event_summary['no_face']]),
        'total_drowsiness_time': sum([e.duration or 0 for e in event_summary['drowsiness_detected']])
    }
    
    return render_template('report.html', 
                         interview=interview,
                         events=events,
                         event_summary=event_summary,
                         stats=stats)

@main_bp.route('/download_report/<int:interview_id>')
def download_report(interview_id):
    """Download proctoring report as CSV"""
    interview = Interview.query.get_or_404(interview_id)
    events = ProctorEvent.query.filter_by(interview_id=interview_id).all()
    
    # Create CSV in memory
    output = BytesIO()
    output.write('\ufeff'.encode('utf8'))  # BOM for Excel compatibility
    
    fieldnames = ['Timestamp', 'Event Type', 'Duration (s)', 'Confidence', 'Details']
    
    csv_content = []
    csv_content.append(','.join(fieldnames))
    
    for event in events:
        row = [
            event.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            event.event_type.replace('_', ' ').title(),
            str(event.duration or 0),
            str(event.confidence or 0),
            str(event.get_details())
        ]
        csv_content.append(','.join([f'"{field}"' for field in row]))
    
    # Add summary
    csv_content.append('')
    csv_content.append('SUMMARY')
    csv_content.append(f'Candidate Name,{interview.candidate.name}')
    csv_content.append(f'Interview Duration,{interview.duration_formatted}')
    csv_content.append(f'Integrity Score,{interview.integrity_score}')
    
    output.write('\n'.join(csv_content).encode('utf-8'))
    output.seek(0)
    
    return send_file(
        output,
        mimetype='text/csv',
        as_attachment=True,
        download_name=f'proctoring_report_{interview_id}_{interview.candidate.name}.csv'
    )

@main_bp.route('/dashboard')
def dashboard():
    """Dashboard showing all interviews"""
    interviews = Interview.query.order_by(Interview.start_time.desc()).all()
    return render_template('dashboard.html', interviews=interviews)

@main_bp.route('/resume_interview/<int:interview_id>')
def resume_interview(interview_id):
    """Resume an active interview session"""
    interview = Interview.query.get_or_404(interview_id)
    
    if interview.status != 'active':
        return redirect(url_for('main.generate_report', interview_id=interview_id))
    
    # Set session data
    session['interview_id'] = interview_id
    session['candidate_name'] = interview.candidate.name
    session['interviewer_name'] = interview.interviewer_name
    
    return redirect(url_for('main.interview_room'))

@main_bp.route('/api/interview_status/<int:interview_id>')
def interview_status(interview_id):
    """Get real-time interview status"""
    interview = Interview.query.get_or_404(interview_id)
    
    # Get recent events (last 30 seconds)
    recent_time = datetime.utcnow() - timedelta(seconds=30)
    recent_events = ProctorEvent.query.filter(
        ProctorEvent.interview_id == interview_id,
        ProctorEvent.timestamp >= recent_time
    ).all()
    
    return jsonify({
        'interview_id': interview_id,
        'status': interview.status,
        'duration': interview.duration or 0,
        'integrity_score': interview.integrity_score,
        'recent_events': [
            {
                'type': event.event_type,
                'timestamp': event.timestamp.isoformat(),
                'message': get_event_message({'type': event.event_type})
            }
            for event in recent_events
        ]
    })