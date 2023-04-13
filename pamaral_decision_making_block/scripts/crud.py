# pip install sqlalchemy
# pip install psycopg2

from sqlalchemy import create_engine
from config import DATABASE_URI
from sqlalchemy.orm import sessionmaker
from models import Base, Flag
import yaml

engine = create_engine(DATABASE_URI)

Session = sessionmaker(bind=engine)

def recreate_db():
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)

    s = Session()

    for data in yaml.load_all(open('flags.yaml')):
        flag = Flag(**data)
        s.add(flag)

    s.commit()

    print(s.query(Flag).all())

    s.close()


def get_flags(color1=None, color2=None):
    s = Session()

    if color1 is None:
        flags = s.query(Flag).all()
    
    elif color2 is None:
        flags = s.query(Flag).filter_by(color1=color1).all()
    
    else:
        flags = s.query(Flag).filter_by(color1=color1, color2=color2).all()

    s.close()

    return flags

print(get_flags("red"))