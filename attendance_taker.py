import dlib
import numpy as np
import cv2
import os
import pandas as pd
import time
import logging
import sqlite3
import datetime
import tkinter as tk
from PIL import Image, ImageTk

# Initialize dlib models
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('data/data_dlib/shape_predictor_68_face_landmarks.dat')
face_reco_model = dlib.face_recognition_model_v1("data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")

# Create a connection to the database
conn = sqlite3.connect("attendance.db")
cursor = conn.cursor()
current_date = datetime.datetime.now().strftime("%Y_%m_%d")
table_name = "attendance"
create_table_sql = f"CREATE TABLE IF NOT EXISTS {table_name} (name TEXT, time_in TEXT, time_out TEXT, date DATE, UNIQUE(name, date))"
cursor.execute(create_table_sql)
conn.commit()
conn.close()

class FaceRecognizer:
    def __init__(self, root, camera_frame, details_frame):
        self.root = root
        self.camera_frame = camera_frame
        self.details_frame = details_frame

        self.font = cv2.FONT_ITALIC
        self.start_time = time.time()
        self.frame_start_time = 0
        self.fps = 0
        self.fps_show = 0
        self.frame_cnt = 0

        self.face_features_known_list = []
        self.face_name_known_list = []
        self.last_frame_face_centroid_list = []
        self.current_frame_face_centroid_list = []
        self.last_frame_face_name_list = []
        self.current_frame_face_name_list = []
        self.last_frame_face_cnt = 0
        self.current_frame_face_cnt = 0
        self.reclassify_interval_cnt = 0
        self.reclassify_interval = 10
        self.details_displayed = False
        self.details_timer = None

        self.get_face_database()

    def get_face_database(self):
        if os.path.exists("data/features_all.csv"):
            path_features_known_csv = "data/features_all.csv"
            csv_rd = pd.read_csv(path_features_known_csv, header=None)
            for i in range(csv_rd.shape[0]):
                features_someone_arr = []
                self.face_name_known_list.append(csv_rd.iloc[i][0])
                for j in range(1, 129):
                    if csv_rd.iloc[i][j] == '':
                        features_someone_arr.append('0')
                    else:
                        features_someone_arr.append(csv_rd.iloc[i][j])
                self.face_features_known_list.append(features_someone_arr)
            logging.info("Faces in Database: %d", len(self.face_features_known_list))
        else:
            logging.warning("'features_all.csv' not found!")

    def update_fps(self):
        now = time.time()
        if str(self.start_time).split(".")[0] != str(now).split(".")[0]:
            self.fps_show = self.fps
        self.start_time = now
        self.frame_time = now - self.frame_start_time
        self.fps = 1.0 / self.frame_time
        self.frame_start_time = now

    @staticmethod
    def return_euclidean_distance(feature_1, feature_2):
        feature_1 = np.array(feature_1)
        feature_2 = np.array(feature_2)
        dist = np.sqrt(np.sum(np.square(feature_1 - feature_2)))
        return dist

    def centroid_tracker(self):
        if len(self.current_frame_face_centroid_list) == 0 or len(self.last_frame_face_centroid_list) == 0:
            return
        
        for i in range(len(self.current_frame_face_centroid_list)):
            e_distance_current_frame_person_x_list = []
            for j in range(len(self.last_frame_face_centroid_list)):
                last_current_frame_centroid_e_distance = self.return_euclidean_distance(
                    self.current_frame_face_centroid_list[i], self.last_frame_face_centroid_list[j])

                e_distance_current_frame_person_x_list.append(
                    last_current_frame_centroid_e_distance)

            last_frame_num = e_distance_current_frame_person_x_list.index(
                min(e_distance_current_frame_person_x_list))
            if last_frame_num < len(self.last_frame_face_name_list):
                self.current_frame_face_name_list[i] = self.last_frame_face_name_list[last_frame_num]

    def draw_note(self, img_rd):
        cv2.putText(img_rd, "Attendance log", (20, 40), self.font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img_rd, "Frame:  " + str(self.frame_cnt), (20, 100), self.font, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(img_rd, "FPS:    " + str(self.fps.__round__(2)), (20, 130), self.font, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(img_rd, "Faces:  " + str(self.current_frame_face_cnt), (20, 160), self.font, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(img_rd, "Press Q to quit", (20, 450), self.font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
        
        for i in range(len(self.current_frame_face_name_list)):
            if i < len(self.current_frame_face_centroid_list):  # Check if index is valid
                img_rd = cv2.putText(img_rd, "Face_" + str(i + 1), tuple(
                    [int(self.current_frame_face_centroid_list[i][0]), int(self.current_frame_face_centroid_list[i][1])]),
                                     self.font, 0.8, (255, 190, 0), 1, cv2.LINE_AA)

    def attendance(self, name):
        current_date = datetime.datetime.now().strftime('%Y-%m-%d')
        conn = sqlite3.connect("attendance.db")
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM attendance WHERE name = ? AND date = ?", (name, current_date))
        existing_entry = cursor.fetchone()
        current_time = datetime.datetime.now().strftime('%H:%M:%S')

        if existing_entry:
            cursor.execute("UPDATE attendance SET time_out = ? WHERE name = ? AND date = ?", (current_time, name, current_date))
            conn.commit()
            print(f"{name} marked as present for {current_date} at {current_time} (out)")
        else:
            cursor.execute("INSERT INTO attendance (name, time_in, date) VALUES (?, ?, ?)", (name, current_time, current_date))
            conn.commit()
            print(f"{name} marked as present for {current_date} at {current_time} (in)")

        conn.close()

    def update_user_details(self, name):
        # Clear previous details
        for widget in self.details_frame.winfo_children():
            widget.destroy()

        # Display user details
        if name != "unknown":
            details = f"Recognized: {name}\n\nTimestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            tk.Label(self.details_frame, text=details, font=("Helvetica", 14, "bold"), fg="green").pack(pady=20)
            tk.Label(self.details_frame, text="Attendance logged", font=("Helvetica", 12), fg="blue").pack(pady=10)
            self.details_displayed = True
            if self.details_timer:
                self.root.after_cancel(self.details_timer)
            self.details_timer = self.root.after(2000, self.clear_user_details)  # Clear after 5 seconds

    def clear_user_details(self):
        for widget in self.details_frame.winfo_children():
            widget.destroy()
        self.details_displayed = False

    def process(self):
        cap = cv2.VideoCapture(0)
        last_detection_time = time.time()

        while cap.isOpened():
            self.frame_cnt += 1
            ret, img_rd = cap.read()
            if not ret:
                break

            self.update_fps()

            # Convert from BGR to RGB
            img_rgb = cv2.cvtColor(img_rd, cv2.COLOR_BGR2RGB)
            faces = detector(img_rgb)

            self.current_frame_face_name_list = []
            self.current_frame_face_position_list = []
            self.current_frame_face_centroid_list = []
            self.current_frame_face_cnt = len(faces)

            if len(faces) != 0:
                self.reclassify_interval_cnt += 1
                if self.reclassify_interval_cnt >= self.reclassify_interval:
                    self.reclassify_interval_cnt = 0

                    current_frame_face_feature_list = []
                    for i, d in enumerate(faces):
                        self.current_frame_face_name_list.append("")
                        self.current_frame_face_position_list.append(tuple([d.left(), int(d.bottom() + (d.bottom() - d.top()) / 4)]))
                        self.current_frame_face_centroid_list.append(
                            [int(d.left() + d.right()) / 2, int(d.top() + d.bottom()) / 2])

                    self.centroid_tracker()
                    for i in range(len(faces)):
                        img_rgb = cv2.rectangle(img_rgb, tuple([d.left(), d.top()]), tuple([d.right(), d.bottom()]), (0, 255, 255), 2)
                        img_rgb = cv2.putText(img_rgb, self.current_frame_face_name_list[i], self.current_frame_face_position_list[i], self.font, 0.8, (0, 255, 255), 1, cv2.LINE_AA)

                else:
                    self.current_frame_face_name_list = []
                    self.reclassify_interval_cnt = 0
                    current_frame_face_feature_list = []
                    self.current_frame_face_position_list = []

                    if len(faces) != 0:
                        for i in range(len(faces)):
                            shape = predictor(img_rgb, faces[i])
                            current_frame_face_feature_list.append(face_reco_model.compute_face_descriptor(img_rgb, shape))
                            self.current_frame_face_position_list.append(tuple([faces[i].left(), int(faces[i].bottom() + (faces[i].bottom() - faces[i].top()) / 4)]))
                            self.current_frame_face_centroid_list.append(
                                [int(faces[i].left() + faces[i].right()) / 2, int(faces[i].top() + faces[i].bottom()) / 2])

                        for k in range(len(faces)):
                            current_frame_e_distance_list = []
                            for i in range(len(self.face_features_known_list)):
                                if str(self.face_features_known_list[i][0]) != '0.0':
                                    e_distance_tmp = self.return_euclidean_distance(current_frame_face_feature_list[k], self.face_features_known_list[i])
                                    current_frame_e_distance_list.append(e_distance_tmp)
                                else:
                                    current_frame_e_distance_list.append(999999999)

                            similar_person_num = current_frame_e_distance_list.index(min(current_frame_e_distance_list))
                            if min(current_frame_e_distance_list) < 0.4:
                                face_name = self.face_name_known_list[similar_person_num]
                                self.current_frame_face_name_list.append(face_name)
                                self.attendance(face_name)
                                self.update_user_details(face_name)
                                last_detection_time = time.time()
                            else:
                                self.current_frame_face_name_list.append("")

                        for i in range(len(faces)):
                            img_rgb = cv2.rectangle(img_rgb, tuple([faces[i].left(), faces[i].top()]),
                                                   tuple([faces[i].right(), faces[i].bottom()]), (0, 255, 255), 2)
                            img_rgb = cv2.putText(img_rgb, self.current_frame_face_name_list[i], self.current_frame_face_position_list[i], self.font, 0.8, (0, 255, 255), 1, cv2.LINE_AA)

            self.draw_note(img_rgb)

            # Convert from RGB to PIL Image
            img_pil = Image.fromarray(img_rgb)
            img_tk = ImageTk.PhotoImage(image=img_pil)

            for widget in self.camera_frame.winfo_children():
                widget.destroy()

            camera_label = tk.Label(self.camera_frame, image=img_tk)
            camera_label.img_tk = img_tk
            camera_label.pack()

            self.root.update_idletasks()
            self.root.update()

            # Re-detect faces if 5 seconds have passed
            if not self.details_displayed and time.time() - last_detection_time > 5:
                self.reclassify_interval_cnt = 0
                last_detection_time = time.time()

            if cv2.waitKey(1) & 0xFF == ord('Q'):
                break

        cap.release()
        cv2.destroyAllWindows()

def main():
    root = tk.Tk()
    root.title("Face Recognition System")

    camera_frame = tk.Frame(root)
    camera_frame.pack(side=tk.LEFT, padx=10, pady=10)

    details_frame = tk.Frame(root)
    details_frame.pack(side=tk.RIGHT, padx=10, pady=10)

    face_recognizer = FaceRecognizer(root, camera_frame, details_frame)
    face_recognizer.process()

if __name__ == "__main__":
    main()