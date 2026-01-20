import cv2
import numpy as np
from google import genai
from google.genai import types
import os
import base64
from datetime import datetime

# Configure Gemini API
GEMINI_API_KEY = "YOUR_API_KEY_HERE"  # Replace with your actual API key
client = genai.Client(api_key=GEMINI_API_KEY)


def get_skin_tone_category(avg_color):
    """Classify skin tone based on average RGB values"""
    r, g, b = avg_color

    # Convert to HSV for better skin tone classification
    hsv = cv2.cvtColor(np.uint8([[avg_color]]), cv2.COLOR_BGR2HSV)[0][0]
    h, s, v = hsv

    # Simplified skin tone classification
    if v < 100:
        return "deep"
    elif v < 140:
        return "dark"
    elif v < 170:
        return "medium"
    elif v < 200:
        return "light"
    else:
        return "fair"


def get_best_hair_colors(skin_tone):
    """Return recommended hair colors based on skin tone"""
    recommendations = {
        "fair": ["Platinum Blonde", "Ash Brown", "Honey Blonde", "Light Auburn"],
        "light": ["Golden Blonde", "Caramel", "Chestnut Brown", "Warm Auburn"],
        "medium": ["Rich Brown", "Dark Chocolate", "Burgundy", "Golden Highlights"],
        "dark": ["Jet Black", "Deep Brown", "Dark Auburn", "Blue-Black"],
        "deep": ["Natural Black", "Deep Espresso", "Dark Plum", "Blue Undertones"]
    }
    return recommendations.get(skin_tone, ["Medium Brown"])


def detect_face_and_skin(frame):
    """Detect face and extract skin tone"""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return None, None

    # Get the largest face
    (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])

    # Extract cheek region for skin tone (avoid eyes, nose)
    cheek_region = frame[y + int(h * 0.4):y + int(h * 0.7), x + int(w * 0.2):x + int(w * 0.4)]

    if cheek_region.size == 0:
        return None, None

    # Calculate average skin color
    avg_color = np.mean(cheek_region, axis=(0, 1))

    return (x, y, w, h), avg_color


def generate_ai_image(skin_tone, hair_color, reference_frame):
    """Generate AI image using Gemini's Imagen 3 model"""
    try:
        # Save reference frame temporarily
        temp_path = "temp_reference.jpg"
        cv2.imwrite(temp_path, reference_frame)

        # Read and encode the image
        with open(temp_path, 'rb') as f:
            image_data = f.read()

        # Create prompt for image generation
        prompt = f"""Transform this person's appearance to show them with {hair_color} hair color.
        The person has {skin_tone} skin tone. Keep their facial features identical, only change the hair color.
        Make it look natural and professional, as if they actually have {hair_color} hair.
        Style: photorealistic, natural lighting, professional portrait."""

        print(f"\n{'=' * 50}")
        print("GENERATING AI IMAGE...")
        print(f"{'=' * 50}")
        print(f"Skin Tone: {skin_tone}")
        print(f"Hair Color: {hair_color}")
        print("Please wait...\n")

        # Use Imagen 3 for image generation via Gemini
        response = client.models.generate_images(
            model='imagen-3.0-generate-001',
            prompt=prompt,
            number_of_images=1,
            safety_filter_level="block_some",
            person_generation="allow_adult",
            aspect_ratio="1:1"
        )

        # Save generated image
        if response.generated_images:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"hair_preview_{hair_color.replace(' ', '_')}_{timestamp}.png"

            # Get the image data
            generated_image = response.generated_images[0]

            # Save to file
            with open(output_filename, 'wb') as f:
                f.write(generated_image.image.data)

            print(f"✅ SUCCESS! Image saved as: {output_filename}")
            print(f"{'=' * 50}\n")

            # Display the generated image
            img = cv2.imread(output_filename)
            if img is not None:
                cv2.imshow('AI Generated Hair Preview', img)
                print("Press any key on the preview window to close it...")
                cv2.waitKey(0)
                cv2.destroyWindow('AI Generated Hair Preview')

            return output_filename
        else:
            print("❌ No image was generated. Try again.")
            return None

    except Exception as e:
        print(f"❌ Gemini API Error: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you have Imagen 3 API access enabled")
        print("2. Check your API key is correct")
        print("3. Verify your billing is set up (Imagen requires billing)")
        return None
    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)


def main():
    # Try different camera indices to find the built-in Mac camera
    print("Searching for cameras...")
    camera_found = False

    for camera_index in range(5):  # Try indices 0-4
        cap = cv2.VideoCapture(camera_index)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"Camera {camera_index} found - Testing...")
                # Display which camera for user to choose
                cv2.putText(frame, f"Camera Index: {camera_index}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow(f'Camera {camera_index} Preview', frame)
                print(f"Is this your Mac's built-in camera? (Showing preview)")
                cv2.waitKey(2000)  # Show for 2 seconds
                cv2.destroyAllWindows()

                user_input = input(f"Use Camera {camera_index}? (y/n): ").lower()
                if user_input == 'y':
                    camera_found = True
                    break
                else:
                    cap.release()
        else:
            cap.release()

    if not camera_found:
        print("No camera selected. Defaulting to camera 0...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("No webcam found. Exiting.")
            return

    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("\n" + "=" * 50)
    print("HAIR COLOR ANALYZER - CONTROLS:")
    print("=" * 50)
    print("SPACE BAR - Analyze and get AI recommendation")
    print("Q - Quit")
    print("=" * 50 + "\n")

    analyzing = False
    best_hair_color = None
    skin_tone_category = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        display_frame = frame.copy()

        # Detect face and skin tone
        face_coords, avg_skin_color = detect_face_and_skin(frame)

        if face_coords is not None:
            x, y, w, h = face_coords
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Get skin tone category
            skin_tone_category = get_skin_tone_category(avg_skin_color)
            best_colors = get_best_hair_colors(skin_tone_category)
            best_hair_color = best_colors[0]

            # Display info
            cv2.putText(display_frame, f"Skin Tone: {skin_tone_category}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_frame, f"Best Hair: {best_hair_color}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Show skin color sample
            cv2.rectangle(display_frame, (10, 80), (60, 130),
                          tuple(map(int, avg_skin_color)), -1)
            cv2.rectangle(display_frame, (10, 80), (60, 130), (255, 255, 255), 2)

            # Display all recommendations
            y_pos = 150
            cv2.putText(display_frame, "Recommended Colors:",
                        (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            for i, color in enumerate(best_colors):
                y_pos += 25
                cv2.putText(display_frame, f"{i + 1}. {color}",
                            (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        else:
            cv2.putText(display_frame, "No face detected - position yourself in frame",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Add controls reminder at bottom
        h, w = display_frame.shape[:2]
        cv2.putText(display_frame, "SPACE=Analyze | Q=Quit",
                    (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow('Hair Color Analyzer', display_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == 32:  # Space bar key code
            if best_hair_color and skin_tone_category:
                print(f"\n{'=' * 50}")
                print("ANALYSIS RESULTS")
                print(f"{'=' * 50}")
                print(f"Skin Tone Category: {skin_tone_category.upper()}")
                print(f"Average Skin Color (BGR): {avg_skin_color}")
                print(f"Top Hair Color Match: {best_hair_color}")
                print(f"All Recommendations: {', '.join(best_colors)}")
                print(f"{'=' * 50}\n")

                # Generate AI image
                print("Generating AI styling recommendation...\n")
                generate_ai_image(skin_tone_category, best_hair_color, frame)
            else:
                print("\n⚠️  No face detected! Position your face in the frame and try again.\n")

        elif key == ord('q') or key == ord('Q'):
            print("\nExiting...")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()