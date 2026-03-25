import cv2
import pygame

# Initialize webcam
cap = cv2.VideoCapture(0)

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Initialize pygame mixer
pygame.mixer.init()

# Load short sound effect
sound_effect = pygame.mixer.Sound("rock_music.mp3")  # replace with your file
pygame.mixer.music.set_volume(1.0)
channel = None  # track playback channel

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        center_y = y + h // 2

        # Looking down → play sound
        if center_y > frame.shape[0] // 2 + 50:
            if channel is None or not channel.get_busy():
                channel = sound_effect.play()

        # Looking up → stop sound immediately
        elif center_y < frame.shape[0] // 2 - 50:
            if channel is not None and channel.get_busy():
                channel.stop()

    cv2.imshow("Head Gesture Sound Control", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
