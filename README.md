Hair Color Analyzer with AI
A Python application that analyzes your skin tone through your webcam and recommends the best hair colors for you. Uses Google's Gemini AI to generate realistic previews of how you'd look with different hair colors.

Features
Real-time face detection and skin tone analysis
Personalized hair color recommendations based on your skin tone
AI-generated images showing you with your best hair color match
Simple webcam interface
Requirements
Python 3.7+
Webcam
Google Gemini API key
Installation
Clone or download this project
Install required packages:
bash
pip install google-genai opencv-python numpy
Get a free Gemini API key:
Visit: https://aistudio.google.com/
Sign in with Google
Click "Get API Key" → "Create API Key"
Copy your key
Add your API key to the script:
Open Basics.py
Replace YOUR_API_KEY_HERE with your actual API key
Usage
Run the script:
bash
python Basics.py
Position your face in the webcam frame
Press SPACE to analyze your skin tone and generate AI preview
Press Q to quit
How It Works
Detects your face using OpenCV
Analyzes skin tone from your cheek area
Categorizes skin tone (fair, light, medium, dark, deep)
Recommends 4 best hair colors for your skin tone
Generates AI image preview with Gemini's Imagen model
Troubleshooting
Camera not working?

Try changing cv2.VideoCapture(0) to cv2.VideoCapture(1) in the code
Disable Continuity Camera in Mac System Settings if using iPhone camera
Keys not responding?

Make sure the OpenCV window is selected/clicked
Try pressing keys while the window is in focus
API errors?

Verify your API key is correct
Check you have Imagen API access enabled
Ensure billing is set up (required for image generation)
Output
Generated images are saved as: hair_preview_[color]_[timestamp].png

Credits
Built with OpenCV, NumPy, and Google Gemini AI

License
Free to use for personal projects

