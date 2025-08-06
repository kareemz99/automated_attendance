import cv2
import numpy as np
import face_recognition
import os
from fastapi import FastAPI, File, UploadFile
import json 
import pickle 
from uvicorn import run

app = FastAPI()

path = "videos/"
images = []
student_names = []
name_to_id = {}

mylist = os.listdir(path)
for stu in mylist:
    folderpath = os.path.join(path, stu)
    for img in os.listdir(folderpath):
        if img.endswith('.jpg'):
            curimg = cv2.imread(f'{folderpath}/{img}')
            images.append(curimg)
            name_parts = os.path.splitext(stu)[0].split('_')
            if len(name_parts) == 2:
                name, student_id = name_parts
                student_names.append(name)
                name_to_id[name] = student_id

file_path = "model.pkl"
with open(file_path, 'rb') as file:
    known = pickle.load(file)

@app.post('/upload')
async def upload_file(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    csv = prediction(image)
    return csv

@app.post('/detect/')
async def detect_faces(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    result = process_file(image)
    return result

def process_file(image):
    attended_response = []
    face_locationss = face_recognition.face_locations(image)
    enc_test = face_recognition.face_encodings(image, face_locationss)
    for each, loc in zip(enc_test, face_locationss):
        is_match = face_recognition.compare_faces(known, each)
        distance = face_recognition.face_distance(known, each)
        best_match = np.argmin(distance)
        if best_match < len(student_names) and is_match[best_match]:
            name = student_names[best_match]
            student_id = name_to_id.get(name, "UNKNOWN")
            print(name, student_id)
            attended_response.append({"name": name, "id": student_id})
            attendance(name, student_id)
    return attended_response

def prediction(image):
    attended_response = []
    face_locationss = face_recognition.face_locations(image)
    enc_test = face_recognition.face_encodings(image, face_locationss)
    for each, loc in zip(enc_test, face_locationss):
        is_match = face_recognition.compare_faces(known, each)
        distance = face_recognition.face_distance(known, each)
        best_match = np.argmin(distance)
        if best_match < len(student_names) and is_match[best_match]:
            name = student_names[best_match]
            student_id = name_to_id.get(name, "UNKNOWN")
            print(name, student_id)
            attended_response.append({"name": name, "id": student_id})
            attendance(name, student_id)
    return json.dumps(attended_response)

def attendance(name, student_id):
    csv_path = "attendance.csv"
    if os.path.exists(csv_path):
        with open(csv_path, "r") as file:
            for line in file:
                if name in line:
                    return False
    with open(csv_path, "a") as file:
        file.write(f"{name},{student_id}\n")
        print(f"{name} with ID {student_id} marked present.")
        return True

if __name__ == '__main__':
    run(app, host="127.0.0.1", port=8000)
