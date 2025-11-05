from database import Base,engine
from models import queueData

Base.metadata.create_all(bind=engine)