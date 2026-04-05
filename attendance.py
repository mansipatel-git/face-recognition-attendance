import cv2
import numpy as np
import os
import csv
from datetime import datetime
from keras_facenet import FaceNet

# ==============================
# 🔹 Initialize FaceNet
# ==============================
embedder = FaceNet()

def get_embedding(face):
    face = cv2.resize(face, (160, 160))
    face = np.expand_dims(face, axis=0)
    return embedder.embeddings(face)[0]

# ==============================
# 🔹 Face Detector
# ==============================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# ==============================
# 🔹 Load Dataset (FIXED 🔥)
# ==============================
dataset_path = "dataset"
known_embeddings = []
known_names = []

print("📂 Dataset Files:", os.listdir(dataset_path))

for file in os.listdir(dataset_path):
    path = os.path.join(dataset_path, file)

    if not file.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    img = cv2.imread(path)

    if img is None:
        print(f"❌ Failed to load {file}")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        print(f"❌ No face found in {file}")
        continue

    (x, y, w, h) = faces[0]
    face = img[y:y+h, x:x+w]

    emb = get_embedding(face)

    known_embeddings.append(emb)
    known_names.append(os.path.splitext(file)[0])

print("✅ Dataset Loaded:", known_names)

if len(known_embeddings) == 0:
    print("❌ No valid dataset images!")
    exit()

# ==============================
# 🔹 Load Group Image
# ==============================
group = cv2.imread("group.jpeg")

if group is None:
    print("❌ Group image not found!")
    exit()

gray = cv2.cvtColor(group, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

print(f"👥 Faces detected: {len(faces)}")

present = set()

# ==============================
# 🔹 Recognize Faces
# ==============================
for (x, y, w, h) in faces:
    face = group[y:y+h, x:x+w]

    try:
        emb = get_embedding(face)
    except:
        continue

    distances = [np.linalg.norm(emb - k) for k in known_embeddings]

    if len(distances) == 0:
        continue

    min_idx = np.argmin(distances)

    print("🔍 Distances:", distances)

    # 🔥 IMPROVED THRESHOLD
    if distances[min_idx] < 0.9:
        name = known_names[min_idx]
        present.add(name)

        print(f"✅ Matched: {name} ({distances[min_idx]:.2f})")

        cv2.rectangle(group, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(group, name, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    else:
        print(f"❌ No match ({distances[min_idx]:.2f})")

# ==============================
# 🔹 Save Attendance
# ==============================
date = datetime.now().strftime("%d-%m-%Y")

with open("attendance.csv", "a", newline="") as f:
    writer = csv.writer(f)

    if f.tell() == 0:
        writer.writerow(["Date", "Name", "Status"])

    for name in known_names:
        status = "Present" if name in present else "Absent"
        writer.writerow([date, name, status])

print("✅ Attendance Saved")

# ==============================
# 🔹 Show Result
# ==============================
cv2.imshow("Attendance Result", group)
cv2.waitKey(0)
cv2.destroyAllWindows()