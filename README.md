# Tutedude SDE Assignment - Focus & Object Detection in Video Interviews

AI-powered video proctoring system that detects candidate focus and unauthorized items during online interviews.

## Features

- **Focus Detection**: Alerts when candidate looks away >5 seconds or no face detected >10 seconds
- **Object Detection**: Identifies phones, books, notes, and electronic devices using YOLO/OpenCV
- **Real-time Monitoring**: Live video feed with instant violation alerts
- **Integrity Scoring**: 100-point system with automatic deductions
- **Comprehensive Reports**: Detailed analytics with CSV export
- **Modern UI**: Responsive design with Tutedude branding

## Technology Stack

- **Backend**: Flask (Python)
- **Database**: SQLite with SQLAlchemy
- **Computer Vision**: OpenCV, YOLO object detection
- **Frontend**: HTML5, Bootstrap 5, JavaScript

## Installation & Setup

### Prerequisites
- Python 3.8+
- Webcam
- Modern web browser (Chrome, Firefox, Safari, Edge)

### Quick Start

1. **Clone Repository**
```bash
git clone <repository-url>
cd poctoring
```

2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

3. **Run Application**
```bash
python run.py
```

4. **Access System**
Open browser: `http://localhost:5002`

### Optional: Generate Demo Data
```bash
python scripts/demo_data.py
```

## Usage

1. **Start Interview**: Click "Start Interview Session" and enter details
2. **Grant Permissions**: Allow camera access when prompted
3. **Begin Monitoring**: Click "Start Monitoring" in interview room
4. **View Reports**: Access dashboard for detailed analytics and CSV downloads

## Detection Capabilities

### Focus Detection
- Gaze tracking using eye position analysis
- Focus loss threshold: 5 seconds
- No face threshold: 10 seconds
- Multiple face detection

### Object Detection
- Mobile phones
- Books and paper notes
- Electronic devices (laptops, tablets)
- Real-time YOLO-based detection with OpenCV fallback

## Scoring System

**Base Score**: 100 points

**Deductions**:
- Focus Loss: -2 points
- No Face: -5 points
- Multiple Faces: -10 points
- Phone Detected: -15 points
- Notes/Books: -10 points
- Drowsiness: -8 points

## Project Structure

```
poctoring/
├── app/
│   ├── api/routes.py          # HTTP endpoints
│   ├── core/proctoring_engine.py  # AI detection logic
│   ├── models/models.py       # Database models
│   ├── templates/             # HTML templates
│   └── utils/helpers.py       # Utility functions
├── scripts/demo_data.py       # Sample data generator
├── run.py                     # Application entry point
└── requirements.txt           # Dependencies
```

## API Endpoints

- `GET /` - Landing page
- `POST /start_interview` - Create interview session
- `GET /interview_room` - Interview monitoring interface
- `POST /process_frame` - Process video frame for detection
- `POST /end_interview` - End session and calculate score
- `GET /report/<id>` - View detailed report
- `GET /download_report/<id>` - Download CSV report
- `GET /dashboard` - Analytics dashboard

## Bonus Features Implemented

- ✅ Eye closure/drowsiness detection
- ✅ Real-time alerts for interviewers
- ✅ Audio monitoring framework
- ✅ Professional Flask architecture
- ✅ Modern responsive UI

## Troubleshooting

**Camera Issues**: Check browser permissions and refresh page
**Detection Accuracy**: Ensure good lighting and clear face visibility
**Performance**: Close unnecessary applications for better processing

## System Requirements

- Camera access permissions
- Stable internet connection
- Well-lit environment
- Clear workspace view

---

**Tutedude Technologies** - AI-Powered Interview Proctoring System