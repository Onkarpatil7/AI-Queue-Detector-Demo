from database import Base,engine
from models import queueData

#Run this to create table in database
Base.metadata.create_all(bind=engine)