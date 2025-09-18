#!/usr/bin/env python3
"""
Video Interview Proctoring System
Main application entry point
"""

import os
import sys
from app import create_app, db
from app.utils.helpers import create_tables

def main():
    """Main application entry point"""
    print("🎯 Video Interview Proctoring System")
    print("=" * 50)
    print("Starting the application...")
    
    # Create Flask app
    app = create_app()
    
    # Create database tables
    with app.app_context():
        create_tables()
        print("✅ Database initialized")
    
    print("🚀 Server starting on http://localhost:5002")
    print("📋 Features available:")
    print("   • Real-time face detection")
    print("   • Focus & attention monitoring") 
    print("   • Object detection (phones, books)")
    print("   • Drowsiness detection")
    print("   • Integrity scoring")
    print("   • Comprehensive reporting")
    print("=" * 50)
    
    try:
        app.run(debug=True, host='0.0.0.0', port=5002)
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()