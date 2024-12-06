# Attendance Management System Using Face Recognition

This project implements an **Attendance Management System** using face recognition technology. The system uses **OpenCV** for real-time face detection and recognition and maintains attendance records efficiently in a CSV file. It is equipped with a user-friendly GUI built using **Tkinter**.

## Features

- **Real-Time Face Detection**: Detects faces using a webcam.
- **Face Recognition**: Identifies individuals and marks their attendance.
- **Attendance Records**: Automatically saves attendance records with timestamps in a CSV file.
- **Training Module**: Includes a feature to train the system with new faces.
- **User-Friendly Interface**: Intuitive GUI for ease of use.

## Requirements

- **Python 3.x**
- **OpenCV** 
- **NumPy**
- **Pandas**

Ensure the necessary packages are installed before running the system.

## Installation

### Step 1: Clone the Repository
Clone the repository to your local machine:
```bash
git clone https://github.com/aadyas17/attendance-management-system.git
cd attendance-management-system
```

### Step 2: Set Up a Virtual Environment
Create and activate a virtual environment to manage dependencies:
```bash
python -m venv venv
# For Windows:
.\venv\Scripts\activate
# For macOS/Linux:
source venv/bin/activate
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
- Launch the main application to start detecting and recognizing faces:
  ```bash
  python main_run.py
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

---

## License

This project is licensed under the [MIT License](LICENSE). Feel free to use and modify it for personal or educational purposes.
