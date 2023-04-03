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


def get_flags():
    s = Session()

    flags = s.query(Flag).all()

    s.close()

    return flags
