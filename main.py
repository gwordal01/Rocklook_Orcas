import cv2
import mediapipe as mp
import pygame
import os
import sys

SUPPORTED = ('.mp3', '.wav', '.ogg')
music_files = [f for f in os.listdir('.') if f.endswith(SUPPORTED)]

if not music_files:
    print("No music files found in this folder.")
    sys.exit(1)

print("\n🎵 Rocklook Playlist 🎵")
for i, track in enumerate(music_files, 1):
    print(f"{i}. {track}")

choice = input("\nSelect track number: ")
try:
    track_index = int(choice) - 1
    MUSIC_FILE = music_files[track_index]
except (ValueError, IndexError):
    print("Invalid choice. Exiting.")
    sys.exit(1)

print(f"\nLoaded: {MUSIC_FILE}")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: No webcam found.")
    sys.exit(1)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

pygame.mixer.init()
pygame.mixer.music.load(MUSIC_FILE)

LEFT_IRIS, RIGHT_IRIS, NOSE_TIP = 468, 473, 1
GAZE_THRESHOLD = -0.12
is_playing = False
was_down = False
print("\nATTENTION: Rocklook started! Look DOWN to play, UP to pause. Press 'q' to quit.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        lm = results.multi_face_landmarks[0].landmark
        iris_y = (lm[LEFT_IRIS].y + lm[RIGHT_IRIS].y) / 2
        nose_y = lm[NOSE_TIP].y
        offset = iris_y - nose_y

        looking_down = offset < GAZE_THRESHOLD
        status = "DOWN !!!" if looking_down else "UP !!!"
        color = (0, 0, 255) if looking_down else (0, 255, 0)

        cv2.putText(frame, status, (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

        if looking_down and not was_down:
            if not is_playing:
                pygame.mixer.music.play(-1)
                is_playing = True
                print("▶ Music started")
            else:
                pygame.mixer.music.unpause()
                print("▶ Music unpaused")

        elif not looking_down and was_down:
            if is_playing:
                pygame.mixer.music.pause()
                print("⏸ Music paused")

        was_down = looking_down
    else:
        cv2.putText(frame, "No face detected", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("Rocklook by Gwordal - Enjoy!", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
print("\nRocklook ended. Come again anytime! 🎸")
