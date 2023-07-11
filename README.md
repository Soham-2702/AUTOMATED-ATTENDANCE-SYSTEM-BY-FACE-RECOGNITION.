ABSTRACT


The main purpose of this project is to build a face recognition-based attendance monitoring system for educational institution to enhance and upgrade the current attendance system into more efficient and effective as compared to before.  The old system has a lot of ambiguity that caused inaccurate and inefficient of attendance taking attendance. Many problems arise when the authority is unable to enforce the regulation that exist in the old system.
The technology working behind will be the face recognition system. The human face is one of the natural traits that can uniquely identify an individual. Therefore, it is used to trace identity as the possibilities for a face to deviate or being duplicated is low. In this project, face databases will be created to pump data into the recognizer algorithm. Then, during the attendance taking session, faces will be compared against the database to seek for identity.
When an individual is identified, its attendance will be taken down automatically saving necessary information into a excel sheet.
Keywords- Smart Attendance System, Face recognition, OpenCV, NumPy



INTRODUCTION

•	The project on Automated Attendance system by face recognition is depend on Computer vision


•	Computer vision is a field of artificial intelligence (AI) that enables computers and systems to derive meaningful information derive meaningful information from digital images, videos and other visual inputs and take actions or make recommendations based on that information.

•	The terminology that we are used to do these projects are ---

Python, Open cv, Face recognition, Os, NumPy, Csv file 


•	Open cv is a python library that allows a human being to perform image processing and computer vision tasks. 

•	It provides a wide range of features, including object detection, face recognition, and tracking.




PROBLEM DEFINITION

•	In this project we have implemented the automated attendance system using FACE RECOGNITION, our working domain.


•	We have projected our ideas to implement “Automated Attendance System Based on Facial Recognition”, in which it imbibes large applications. The application includes face identification, which saves time and eliminates chances of proxy attendance because of the face authorization. Hence, this system can be implemented in a field where attendance plays an important role. 


•	The system is designed using OPEN CV platform. The library that  system uses  was proposed by HOG method(Histogram Of Gradient) and  uses a well trained residual Neural Network(RNN), a type of neural network .


•	This algorithm compares the test image and training image and determines students who are present and absent. The attendance record is maintained in an excel sheet which is updated automatically in the system.



ARCHITECTURE

face_recognition.api.batch_face_locations(images, number_of_times_to_upsample=1, batch_size=128)

Returns an 2d array of bounding boxes of human faces in a image using the cnn face detector If you are using a GPU, this can give you much faster results since the GPU can process batches of images at once. If you aren’t using a GPU, you don’t need this function.
face_recognition.api.compare_faces(known_face_encodings, face_encoding_to_check, tolerance=0.6)

Compare a list of face encodings against a candidate encoding to see if they match.
face_recognition.api.face_distance(face_encodings, face_to_compare)

Given a list of face encodings, compare them to a known face encoding and get a euclidean distance for each comparison face. The distance tells you how similar the faces are.
face_recognition.api.face_encodings(face_image, known_face_locations=None, num_jitters=1, model='small')

Given an image, return the 128-dimension face encoding for each face in the image.
128 landmarks extracted from each face
face_recognition.api.face_landmarks(face_image, face_locations=None, model='large')

Given an image, returns a dict of face feature locations (eyes, nose, etc) for each face in the image
face_recognition.api.face_locations(img, number_of_times_to_upsample=1, model='hog')
Returns an array of bounding boxes of human faces in a image

face_recognition.api. load_image_file (file, mode='RGB')
Loads an image file (.jpg, .png, etc) into a numpy array.




DETAILED WORKING

Steps of Working:
- Initiate the OPEN CV python script.
- Create a DATASET of the student by entering their photos.
- We have done these encodings of these raw data of the faces and stores as a “_encoding”.
- Then we have created a list both for the encoding and for names.  The “known faces” file is created to train the dataset.
- we have created an infinite loop using while true, then we are reading the video input. And then we are decreasing the size of the input that we are getting from the webcam and then we are converting it in rgb .
- A picture of the class is taken, and the FACE RECOGNITION python file is initiated.
- Attendance is taken by cropping the faces in the picture and comparing with the faces in the database.
- If a face is matched, the responding name is marked in an EXCEL file with the current date and time.

