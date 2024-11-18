**Face Recognition Logging System**

The Face Recognition Logging System is an application designed for real-time facial recognition and attendance logging. It uses a webcam to capture, recognize, and log user details into a local SQLite database. The system provides tools to register new users, capture their facial data, extract facial features, and log attendance efficiently.


Features
- Face Registration: Easily register new users with photos for improved recognition accuracy.
- Feature Extraction: Converts captured images into feature vectors for reliable facial recognition.
- Attendance Logging: Automatically logs recognized users into an SQLite database with timestamps.
- Optimized for Raspberry Pi: Designed for lightweight operation on Raspberry Pi setups.


Download Additional Data

Download the required data from the following Google Drive link:
https://drive.google.com/drive/folders/1os37oXbump6CTKDThdxZcWpTIcZqcTJS?usp=drive_link

Place the extracted files in the data/ folder of the project directory:


**How It Works**

Step 1: Register a New User

Run the register.py file:
bash
Copy code
python register.py  
Enter the user's name and employee number in the provided fields.
Click Submit.
This creates a temporary folder for the user in the data_faces_from_camera/ directory.
Use the webcam to capture around 50 photos of the user's face in different angles (e.g., with and without glasses, varying expressions).
The more photos you capture, the more accurate the system will be.

Step 2: Extract Facial Features

After capturing the photos, click Extract Feature.
The system will process the images and save the associated feature vectors in a CSV file.
Once the feature extraction is complete, the captured images will automatically be deleted upon closing the program.

Step 3: Take Attendance

Run the attendance_taker.py file:
 
The system will recognize users in real time via the webcam and log their attendance into the SQLite database, including timestamps.
