# Attendance Management System Using Face Recognition

This project is an advanced Attendance Management System that utilizes face recognition technology for efficient attendance tracking. It leverages OpenCV for real-time face detection and recognition.

## Features

- Detects faces in real-time using a webcam.
- Recognizes faces and marks attendance.
- Stores attendance records in a CSV file.
  
## Requirements

- **Python**
- **OpenCV** 
- **NumPy**
- **Pandas**
- **Pillow**
- **Tkinter** 

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

Tkinter (Desktop Application)

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
├── datasets/                # Directory containing training images
├── [training.py](http://_vscodecontentref_/1)              # Script to train the face recognition model
├── main_run_tkinter.py      # Main script to run the attendance system with Tkinter
├── [requirements.txt](http://_vscodecontentref_/3)         # List of required packages
├── attendance.csv           # Sample attendance record file (optional)
├── .gitignore               # Git ignore file
├── .gitattributes           # Git attributes file (optional)
├── README.md                # Project documentation
└── LICENSE                  # License file (optional)
```
