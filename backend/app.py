from fastapi import FastAPI, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session
from datetime import datetime, timezone
from database import sessionLocal
from models import queueData 

app = FastAPI()

class DetectionData(BaseModel):
    id: int           
    waitTime: float   
def get_db():
    db = sessionLocal()  # Create new session
    try:
        yield db
    finally:
        db.close()

@app.get("/")
def root():
    return {"message":"backend is running"}

@app.post("/update_data/")
def update_detection(data: DetectionData, db: Session = Depends(get_db)):
    """
    This route receives the detected person's ID and waitTime from YOLO script
    and saves them into the PostgreSQL database with a timestamp.
    """
    entry = queueData(
        id=data.id,                              #Use YOLO's person ID
        timeStamp=datetime.now(timezone.utc),    
        waitTime=data.waitTime                   # Save the person's waiting time
    )

    db.add(entry)
    db.commit()
    db.refresh(entry)  # Refresh to get any auto-generated fields

    return {
        "message": "Data saved successfully",
        "id": entry.id,
        "timeStamp": entry.timeStamp.isoformat(),
        "waitTime": entry.waitTime
    }
