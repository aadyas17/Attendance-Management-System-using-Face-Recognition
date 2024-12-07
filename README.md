# Attendance Management System Using Face Recognition

This project is an Attendance Management System that uses face recognition to mark attendance. It leverages OpenCV for face detection and recognition.

## Features

- Detects faces in real-time using a webcam.
- Recognizes faces and marks attendance.
- Stores attendance records in a CSV file.
- Provides both a web-based interface (Streamlit) and a desktop application (Tkinter)
- 
## Requirements

- **Python 3.x**
- **OpenCV** 
- **NumPy**
- **Pandas**
- **Pillow**
- **Streamlit** (for web interface)
- **Tkinter** (for desktop application)

Ensure the necessary packages are installed before running the system.

## Installation

### Step 1: Clone the Repository
1. **Clone the repository**:
   ```sh
   git clone https://github.com/aadyas17/attendance-management-system.git
   cd attendance-management-system

### Step 2: Create a virtual environment and activate it:
```bash
python -m venv venv
.\venv\Scripts\activate
```

### Step 3: Install Required Packages
Install all required packages:
```bash
pip install -r requirements.txt
```

## Usage

### 1. **Training the Model**
- Place training images in the `dataset` directory. Each image should follow the naming format:
  ```
  User.<ID>.<ImageNumber>.jpg
  ```

- Run the training script to train the face recognition model:
  ```bash
  python training.py
  ```

### 2. **Running the Attendance System**
Running the Attendance System

1.Streamlit (Web Interface)

Run the Streamlit script:
```bash
streamlit run main_run_streamlit.py
```

2. Tkinter (Desktop Application)

Run the Tkinter script:
```bash
python main_Run.py
```

### 3. **Viewing Attendance Records**
- Attendance records are stored in the `attendance/attendance.csv` file. Open this file with any text editor or spreadsheet software to view the attendance log.

---

## Project Structure

```
attendance-management-system/
│
├── dataset/                 # Directory containing training images
├── attendance/              # Directory containing attendance logs
├── trainer/                 # Directory for trained model files
├── main_run.py              # Main script to run the attendance system
├── training.py              # Script to train the face recognition model
├── requirements.txt         # List of required Python packages
├── README.md                # Project documentation
└── ...                      # Other scripts and files
```
