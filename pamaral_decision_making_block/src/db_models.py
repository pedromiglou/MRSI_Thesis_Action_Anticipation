from sqlalchemy import Column, Integer, String, Float
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
        return "<Flag(country='{}', color1='{}', color2={}, color3={})>" \
            .format(self.country, self.color1, self.color2, self.color3)


class Probability(Base):
    __tablename__ = 'Probability'
    id = Column(Integer, primary_key=True)
    color1 = Column(String)
    color2 = Column(String)
    color3 = Column(String)
    value = Column(Float)

    def __repr__(self):
        return "<Probability(color1='{}', color2={}, color3={}, value={})>" \
            .format(self.color1, self.color2, self.color3, self.value)
