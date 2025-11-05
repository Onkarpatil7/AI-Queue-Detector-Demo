from sqlalchemy import Column,Integer,Float,String,DateTime
from database import Base
from datetime import datetime

class queueData(Base):
    __tablename__="queuedata"

    id=Column(Integer,primary_key=True,index=True)
    timeStamp=Column(DateTime,default=datetime.utcnow)
    peopleCount=Column(Integer)
    waitTime=Column(Integer)

    