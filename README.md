# AI Queue Detector

An intelligent queue management system that uses YOLOv8 for real-time person detection and tracking. The system monitors entry/exit events, calculates wait times, and stores all data in a PostgreSQL database.

## Technology Stack

- **Backend**: FastAPI (Python)
- **Database**: PostgreSQL
- **AI/ML**: YOLOv8 (Ultralytics)
- **Computer Vision**: OpenCV

## Installation Steps

### 1. Install Dependencies

Navigate to the `backend` directory and install the required packages:

```bash
cd backend
pip install -r requirements.txt
```

### 2. Set Up PostgreSQL Database

1. **Install PostgreSQL** (if not already installed)
   - Download from [PostgreSQL Official Website](https://www.postgresql.org/download/)

2. **Create a Database**
   ```sql
   CREATE DATABASE queue_detector;
   ```

### 3. Configure Environment Variables

Create a `.env` file in the `backend` directory with your PostgreSQL database configuration:

```env
DB_HOST=localhost
DB_NAME=queue_detector
DB_USER=your_username
DB_PASSWORD=your_password
DB_PORT=5432
```

### 4. Create Database Tables

Run the `create.py` script **once** to create the necessary database tables:

```bash
cd backend
python create.py
```

## Running the Application

### Step 1: Start the Backend Server

Open a terminal and navigate to the `backend` directory:

```bash
cd backend
uvicorn app:app --reload
```

The FastAPI server will start on `http://localhost:8000`

### Step 2: Run Detection Script

Open a **new terminal** and navigate to the `backend` directory:

```bash
cd backend
python detection.py
```

This will start the camera feed and begin detecting and tracking people. Press `Q` to quit.
