from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class Flag(Base):
    __tablename__ = 'Flag'
    id = Column(Integer, primary_key=True)
    country = Column(String)
    color1 = Column(String)
    color2 = Column(String)
    color3 = Column(String)
    
    def __repr__(self):
        return "<Flag(country='{}', color1='{}', color2={}, color3={})>"\
                .format(self.country, self.color1, self.color2, self.color3)
