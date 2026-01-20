# Hair Color Analyzer with AI

AI-powered webcam app that analyzes your skin tone and recommends the best hair colors for you. Uses Google Gemini's Imagen 3 to generate realistic previews of how you'd look with different hair colors.

## Features

- Real-time face detection and skin tone analysis
- Personalized hair color recommendations based on your skin tone
- AI-generated images showing you with your best hair color match
- Easy-to-use webcam interface

## Requirements

- Python 3.7+
- Webcam
- Google Gemini API key (free tier available)

## Installation

1. Install required packages:
```bash
pip install google-genai opencv-python numpy
```

2. Get a free Gemini API key:
   - Go to https://aistudio.google.com/
   - Click "Get API Key"
   - Copy your key

3. Add your API key to the script:
```python
GEMINI_API_KEY = "YOUR_API_KEY_HERE"  # Replace with your actual key
```

## Usage

1. Run the script:
```bash
python Basics.py
```

2. Position your face in the webcam frame

3. Press **SPACEBAR** to analyze and generate AI preview

4. Press **Q** to quit

## Controls

- **SPACE** - Analyze skin tone and generate AI hair preview
- **Q** - Quit application

## Output

Generated images are saved as:
```
hair_preview_[ColorName]_[timestamp].png
```

## Troubleshooting

**Wrong camera opening (iPhone instead of Mac)?**
- Disable Continuity Camera in System Settings
- Or try changing camera index: `cv2.VideoCapture(1)`

**No face detected?**
- Ensure good lighting
- Move closer to camera
- Face the camera directly

**API errors?**
- Check your API key is correct
- Verify internet connection
- Enable Vertex AI and Imagen API in Google Cloud Console

## Note

Imagen 3 image generation requires a Google Cloud billing account. Free tier includes $300 credits.

## License

Free to use for personal projects.
