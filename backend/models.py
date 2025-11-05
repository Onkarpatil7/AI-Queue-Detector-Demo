from sqlalchemy import Column,Integer,Float,String,DateTime
from database import Base
from datetime import datetime

class queueData(Base):
    __tablename__="queuedata"


    timeStamp=Column(DateTime,default=datetime.now,primary_key=True)
    peopleCount=Column(Integer)
    waitTime=Column(Integer)

