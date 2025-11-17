# AI-Based Face Recognition Attendance System

A real-time face recognition attendance system built with Python, OpenCV, and Machine Learning. The system captures faces through a webcam, recognizes registered users, and automatically logs their attendance with timestamps.

## Features

- **Real-time Face Detection**: Uses Haar Cascade classifier for accurate face detection
- **Face Recognition**: Implements the K-Nearest Neighbors (KNN) algorithm for face recognition
- **Automated Attendance Logging**: Records attendance with date and time stamps in CSV format
- **Voice Feedback**: Provides audio confirmation when attendance is marked
- **Multiple User Support**: Can register and recognize multiple users
- **Custom Background**: Displays video feed with a custom background image
- **Web Dashboard**: Streamlit-based web interface to view attendance records

## Tech Stack:

- **Python 3.x**
- **OpenCV**: For image processing and face detection
- **scikit-learn**: KNN classifier for face recognition
- **NumPy**: Array operations
- **Pandas**: Data manipulation
- **Streamlit**: Web dashboard for viewing attendance
- **win32com**: Voice feedback system

## Project Structure

```
Face_Recognition_Based_Attendance/
│
├── Data/
│   ├── haarcascade_frontalface_default.xml
│   ├── names.pkl
│   └── faces_data.pkl
│
├── Attendance/
│   └── Attendance_DD-MM-YYYY.csv
│
├── add_faces.py          # Register new users
├── test.py               # Main attendance system
├── app.py                # Streamlit dashboard
├── background.png        # UI background image
└── README.md
```

## Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/Face_Recognition_Based_Attendance.git
cd Face_Recognition_Based_Attendance
```

2. **Install required packages**
```bash
pip install opencv-python
pip install scikit-learn
pip install numpy
pip install pandas
pip install streamlit
pip install streamlit-autorefresh
pip install pywin32
```

3. **Create necessary folders**
```bash
mkdir Data
mkdir Attendance
```

4. **Download Haar Cascade file**
- Download `haarcascade_frontalface_default.xml` from [OpenCV GitHub](https://github.com/opencv/opencv/tree/master/data/haarcascades)
- Place it in the `Data/` folder

## Usage

### 1. Register New Users

Run the face registration script:

```bash
python add_faces.py
```

- Enter your name when prompted
- Look at the camera and let it capture 100 images of your face
- Press 'q' to quit early if needed
- The system saves your face data automatically

### 2. Run Attendance System

Start the face recognition system:

```bash
python test.py
```

- The camera will open and start detecting faces
- When your face is recognized, your name will appear on screen
- Press 'o' to mark attendance (voice confirmation will play)
- Press 'q' to quit the system

### 3. View Attendance Records

Launch the web dashboard:

```bash
python -m streamlit run app.py
```

- Opens in your browser automatically
- Displays today's attendance in a table
- Auto-refreshes every 2 seconds

## How It Works

1. **Face Registration (`add_faces.py`)**:
   - Captures 100 images of a person's face
   - Converts images to grayscale (50x50 pixels)
   - Flattens each image into a 2500-feature vector
   - Stores face data and names in pickle files

2. **Face Recognition (`test.py`)**:
   - Loads trained face data
   - Trains KNN classifier with stored faces
   - Detects faces in real-time from webcam
   - Predicts identity using KNN
   - Logs attendance when 'o' key is pressed

3. **Attendance Logging**:
   - Creates CSV files named `Attendance_DD-MM-YYYY.csv`
   - Records name and timestamp for each entry
   - Prevents duplicate entries in the same file

## Troubleshooting

### Camera Not Opening
- Check if another application is using the camera
- Try changing camera index: `cv2.VideoCapture(1)` instead of `(0)`

### Face Not Detected
- Ensure good lighting conditions
- Adjust `detectMultiScale` parameters
- Check if Haar Cascade file exists in `Data/` folder

### Recognition Accuracy Issues
- Register more face samples per person
- Ensure consistent lighting during registration and recognition
- Keep face clearly visible and avoid obstructions

### Size Mismatch Errors
- Delete `Data/names.pkl` and `Data/faces_data.pkl`
- Re-register all users with consistent settings

## Future Enhancements

- [ ] Add deep learning models (CNN) for better accuracy
- [ ] Support for multiple cameras
- [ ] Email notifications for attendance
- [ ] Admin panel for managing users
- [ ] Export attendance reports to Excel
- [ ] Mobile app integration
- [ ] Cloud storage for attendance data

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Acknowledgments

- OpenCV library for computer vision functionality
- scikit-learn for machine learning algorithms
- Streamlit for easy web dashboard creation

## Contact

Contact - [Email Me](shantanupanchal.dev@gmail.com)

Project Link: [https://github.com/Shantanu-0101/AI-Based_Attendance_System](https://github.com/Shantanu-0101/AI-Based_Attendance_System)

---

**Note**: This system is designed for educational purposes. For production use, consider implementing additional security measures and using more advanced face recognition models.
