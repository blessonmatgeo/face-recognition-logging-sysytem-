import dlib
import numpy as np
import cv2
import os
import shutil
import time
import logging
import tkinter as tk
from tkinter import font as tkFont
from PIL import Image, ImageTk
import atexit
import csv
import numpy as np
import logging
from tkinter import filedialog

detector = dlib.get_frontal_face_detector()
path_images_from_camera = "data/data_faces_from_camera/"
predictor = dlib.shape_predictor('data/data_dlib/shape_predictor_68_face_landmarks.dat')
face_reco_model = dlib.face_recognition_model_v1("data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")

class Face_Register:
    def __init__(self):
        self.current_frame_faces_cnt = 0  
        self.existing_faces_cnt = 0  
        self.ss_cnt = 0  

        self.win = tk.Tk()
        self.win.title("Face Register")
        self.win.geometry("1100x950")

        self.frame_left_camera = tk.Frame(self.win)
        self.label = tk.Label(self.win)
        self.label.pack(side=tk.LEFT)
        self.frame_left_camera.pack()

        self.frame_right_info = tk.Frame(self.win)
        self.label_cnt_face_in_database = tk.Label(self.frame_right_info, text=str(self.existing_faces_cnt))
        self.label_fps_info = tk.Label(self.frame_right_info, text="")
        self.input_name = tk.Entry(self.frame_right_info)
        self.input_name_char = ""
        self.input_employee_number = tk.Entry(self.frame_right_info) # Employee number input
        self.label_warning = tk.Label(self.frame_right_info)
        self.label_face_cnt = tk.Label(self.frame_right_info, text="Faces in current frame: ")
        self.log_all = tk.Label(self.frame_right_info)

        self.font_title = tkFont.Font(family='Helvetica', size=20, weight='bold')
        self.font_step_title = tkFont.Font(family='Helvetica', size=15, weight='bold')
        self.font_warning = tkFont.Font(family='Helvetica', size=15, weight='bold')

        self.path_photos_from_camera = "data/data_faces_from_camera/"
        self.current_face_dir = ""
        self.font = cv2.FONT_ITALIC

        self.extract_button = tk.Button(
            self.frame_right_info,
            text='Extract Features',
            command=self.extract_features
        )
        self.extract_button.grid(row=20, column=0, columnspan=3, pady=10)

        self.current_frame = np.ndarray
        self.face_ROI_image = np.ndarray
        self.face_ROI_width_start = 0
        self.face_ROI_height_start = 0
        self.face_ROI_width = 0
        self.face_ROI_height = 0
        self.ww = 0
        self.hh = 0

        self.out_of_range_flag = False
        self.face_folder_created_flag = False

        self.cap = cv2.VideoCapture(0)  

    def GUI_get_input_name(self):
        self.input_name_char = self.input_name.get()
        self.input_employee_number_char = self.input_employee_number.get() # Get employee number
        self.create_face_folder()
        self.label_cnt_face_in_database['text'] = str(self.existing_faces_cnt)

    def GUI_info(self):
        tk.Label(self.frame_right_info,
                 text="Face register",
                 font=self.font_title).grid(row=0, column=0, columnspan=3, sticky=tk.W, padx=2, pady=20)

        tk.Label(self.frame_right_info,
                 text="Faces in current frame: ").grid(row=3, column=0, columnspan=2, sticky=tk.W, padx=5, pady=2)
        self.label_face_cnt.grid(row=3, column=2, columnspan=3, sticky=tk.W, padx=5, pady=2)

        self.label_warning.grid(row=4, column=0, columnspan=3, sticky=tk.W, padx=5, pady=2)

        tk.Label(self.frame_right_info,
                 font=self.font_step_title,
                 text="Enter your name").grid(row=7, column=0, columnspan=2, sticky=tk.W, padx=5, pady=20)

        tk.Label(self.frame_right_info, text="Name: ").grid(row=8, column=0, sticky=tk.W, padx=5, pady=0)
        self.input_name.grid(row=8, column=1, sticky=tk.W, padx=0, pady=2)

        tk.Label(self.frame_right_info, text="Employee Number: ").grid(row=9, column=0, sticky=tk.W, padx=5, pady=0) # Employee number label
        self.input_employee_number.grid(row=9, column=1, sticky=tk.W, padx=0, pady=2) # Employee number input field

        tk.Button(self.frame_right_info,
                 text='Submit',
                 command=self.GUI_get_input_name).grid(row=8, column=2, padx=5)

        tk.Label(self.frame_right_info,
                 font=self.font_step_title,
                 text="Save atleast 7 different angles of your face ").grid(row=12, column=0, columnspan=2, sticky=tk.W, padx=5, pady=20)

        tk.Button(self.frame_right_info,
                 text='Save current face',
                 command=self.save_current_face).grid(row=15, column=0, columnspan=3, sticky=tk.W)

        self.log_all.grid(row=11, column=0, columnspan=20, sticky=tk.W, padx=5, pady=20)

        self.frame_right_info.pack()

    def pre_work_mkdir(self):
        if os.path.isdir(self.path_photos_from_camera):
            pass
        else:
            os.mkdir(self.path_photos_from_camera)

    def create_face_folder(self):
        self.existing_faces_cnt += 1
        if self.input_name_char and self.input_employee_number_char:
            self.current_face_dir = os.path.join(self.path_photos_from_camera, f"{self.input_employee_number_char}_{self.input_name_char}")
        else:
            self.current_face_dir = os.path.join(self.path_photos_from_camera, f"person_{self.existing_faces_cnt}")
        os.makedirs(self.current_face_dir)
        self.log_all["text"] = f"\"{self.current_face_dir}/\" created!"
        logging.info("\n%-40s %s", "Create folders:", self.current_face_dir)

        self.ss_cnt = 0  
        self.face_folder_created_flag = True   


    def save_current_face(self):
        if self.face_folder_created_flag:
            if self.current_frame_faces_cnt == 1:
                if not self.out_of_range_flag:
                    self.ss_cnt += 1
                    self.face_ROI_image = np.zeros((int(self.face_ROI_height * 2), self.face_ROI_width * 2, 3),
                                                   np.uint8)
                    for ii in range(self.face_ROI_height * 2):
                        for jj in range(self.face_ROI_width * 2):
                            self.face_ROI_image[ii][jj] = self.current_frame[self.face_ROI_height_start - self.hh + ii][
                                self.face_ROI_width_start - self.ww + jj]
                    self.log_all["text"] = "\"" + self.current_face_dir + "/img_face_" + str(
                        self.ss_cnt) + ".jpg\"" + " saved!"
                    self.face_ROI_image = cv2.cvtColor(self.face_ROI_image, cv2.COLOR_BGR2RGB)

                    cv2.imwrite(self.current_face_dir + "/img_face_" + str(self.ss_cnt) + ".jpg", self.face_ROI_image)
                    logging.info("%-40s %s/img_face_%s.jpg", "Save into：",
                                 str(self.current_face_dir), str(self.ss_cnt) + ".jpg")
                else:
                    self.log_all["text"] = "Please do not go out of range!"
            else:
                self.log_all["text"] = "No face in current frame!"

    def get_frame(self):
        try:
            if self.cap.isOpened():
                ret, frame = self.cap.read()
                frame = cv2.resize(frame, (640,480))
                return ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except:
            print("Error: No video input!!!")

    #Face detection and saving
    def process(self):
        ret, self.current_frame = self.get_frame()
        faces = detector(self.current_frame, 0)
        if ret:
            self.label_face_cnt["text"] = str(len(faces))
            if len(faces) != 0:
                for k, d in enumerate(faces):
                    self.face_ROI_width_start = d.left()
                    self.face_ROI_height_start = d.top()
                    self.face_ROI_height = (d.bottom() - d.top())
                    self.face_ROI_width = (d.right() - d.left())
                    self.hh = int(self.face_ROI_height / 2)
                    self.ww = int(self.face_ROI_width / 2)

 
                    if (d.right() + self.ww) > 640 or (d.bottom() + self.hh > 480) or (d.left() - self.ww < 0) or (
                            d.top() - self.hh < 0):
                        self.label_warning["text"] = "OUT OF RANGE, Please readjust your face!"
                        self.label_warning['fg'] = 'red'
                        self.out_of_range_flag = True
                        color_rectangle = (255, 0, 0)
                    else:
                        self.out_of_range_flag = False
                        self.label_warning["text"] = "Ready to capture"
                        color_rectangle = (0, 255,0 )
                    self.current_frame = cv2.rectangle(self.current_frame,
                                                       tuple([d.left() - self.ww, d.top() - self.hh]),
                                                       tuple([d.right() + self.ww, d.bottom() + self.hh]),
                                                       color_rectangle, 2)
            self.current_frame_faces_cnt = len(faces)

            img_Image = Image.fromarray(self.current_frame)
            img_PhotoImage = ImageTk.PhotoImage(image=img_Image)
            self.label.img_tk = img_PhotoImage
            self.label.configure(image=img_PhotoImage)

        # Refresh frame
        self.win.after(20, self.process)

    def run(self):
        self.pre_work_mkdir()
        self.GUI_info()
        self.process()
        self.win.mainloop()
        atexit.register(self.remove_created_folder)

    def extract_features(self):
        path_images_from_camera = "data/data_faces_from_camera/"
        try:
            self.extract_and_save_features(path_images_from_camera)
            self.log_all["text"] = "Feature extraction completed!"
        except Exception as e:
            self.log_all["text"] = f"Error during feature extraction: {str(e)}"

    def remove_created_folder(self):
        if self.face_folder_created_flag and os.path.exists(self.current_face_dir):
            shutil.rmtree(self.current_face_dir)
            logging.info("\n%-40s %s", "Remove folders:", self.current_face_dir)

    
    def return_128d_features(self, path_img):
        img_rd = cv2.imread(path_img)
        faces = detector(img_rd, 1)

        logging.info("%-40s %-20s", " Image with faces detected:", path_img)

        if len(faces) != 0:
            shape = predictor(img_rd, faces[0])
            face_descriptor = face_reco_model.compute_face_descriptor(img_rd, shape)
        else:
            face_descriptor = 0
            logging.warning("no face")
        return face_descriptor

    
    def return_features_mean_personX(self, path_face_personX):
        features_list_personX = []

        if not os.path.isdir(path_face_personX):
            logging.warning("Warning: '%s' is not a directory.", path_face_personX)
            return np.zeros(128, dtype=object, order='C')

        photos_list = os.listdir(path_face_personX)
        if photos_list:
            for i in range(len(photos_list)):
                logging.info("%-40s %-20s", " / Reading image:", path_face_personX + "/" + photos_list[i])
                features_128d = self.return_128d_features(path_face_personX + "/" + photos_list[i])

                if features_128d == 0:
                    i += 1
                else:
                    features_list_personX.append(features_128d)
        else:
            logging.warning("Warning: No images in %s/", path_face_personX)

        if features_list_personX:
            features_mean_personX = np.array(features_list_personX, dtype=object).mean(axis=0)
        else:
            features_mean_personX = np.zeros(128, dtype=object, order='C')
        return features_mean_personX

    
    def extract_and_save_features(self, path_images_from_camera):
        logging.basicConfig(level=logging.INFO)
        #  Get the order of the latest person
        person_list = os.listdir("data/data_faces_from_camera/")
        person_list.sort()

        csv_file_path = "data/features_all.csv"

        # Check if the CSV file already exists
        csv_file_exists = os.path.isfile(csv_file_path)

        with open(csv_file_path, mode='a', newline="") as csvfile:
            fieldnames = list(range(1, 129))
            fieldnames.insert(0, 'Employee')

            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            # If the file doesn't exist, write the header
            if not csv_file_exists:
                writer.writeheader()

            for person in person_list:
                # Get the mean/average features of face/personX, it will be a list with a length of 128D
                logging.info("%sperson_%s", path_images_from_camera, person)
                features_mean_personX = self.return_features_mean_personX(path_images_from_camera + person)

                if len(person.split('_', 2)) == 2:
                    person_name = person
                else:
                    person_name = person.split('_', 2)[-1]

                # features_mean_personX will be 129D, person name + 128 features
                row_dict = dict(zip(fieldnames, [person_name] + list(features_mean_personX)))
                writer.writerow(row_dict)
                logging.info('\n')
            logging.info("Saved all the features of faces registered into: data/features_all.csv")

def main():
    logging.basicConfig(level=logging.INFO)
    Face_Register_con = Face_Register()
    Face_Register_con.run()
    
    
if __name__ == '__main__':
    main()
