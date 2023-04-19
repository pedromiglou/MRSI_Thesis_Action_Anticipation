#!/usr/bin/env python3

# pip install sqlalchemy
# pip install psycopg2

import rospy
import yaml

from db_models import Base, Flag, Probability
from pamaral_decision_making_block.srv import GetProbabilities, GetProbabilitiesResponse
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


class Database:
    """
    Class containing methods to communicate with the database.
    """

    def __init__(self):
        # Scheme: "postgres+psycopg2://<USERNAME>:<PASSWORD>@<IP_ADDRESS>:<PORT>/<DATABASE_NAME>"
        self.engine = create_engine('postgresql+psycopg2://postgres:password@localhost:5432/postgres')

        self.Session = sessionmaker(bind=self.engine)

        self.recreate_db()

        rospy.Service("get_probabilities", GetProbabilities, self.get_probabilities)

    def recreate_db(self):
        # recreate all tables
        Base.metadata.drop_all(self.engine)
        Base.metadata.create_all(self.engine)

        s = self.Session()

        # add flags to database
        for data in yaml.load_all(open("/home/miglou/catkin_ws/src/MRSI_Thesis/data/flags.yaml")):
            flag = Flag(**data)
            s.add(flag)

        s.commit()

        # add probabilities to database
        # color1 color2 color3 value
        flags = s.query(Flag).all()

        colors = []
        for f in flags:
            colors.extend([f.color1, f.color2, f.color3])
        colors = set(colors)

        for color1 in colors:
            p = Probability(color1=color1, color2="", color3="",
                            value=len([f for f in flags if f.color1 == color1])/len(flags))
            s.add(p)
            if len([f for f in flags if f.color1 == color1]) != 0:
                for color2 in colors:
                    if color2 != color1:
                        p = Probability(color1=color1, color2=color2, color3="",
                                        value=len([f for f in flags if f.color1 == color1 and f.color2 == color2]) /
                                        len([f for f in flags if f.color1 == color1]))
                        s.add(p)
                        if len([f for f in flags if f.color1 == color1 and f.color2 == color2]) != 0:
                            for color3 in colors:
                                if color3 != color1 and color3 != color2:
                                    p = Probability(color1=color1, color2=color2, color3=color3,
                                                    value=len([f for f in flags if f.color1 == color1 and
                                                               f.color2 == color2 and f.color3 == color3]) /
                                                    len([f for f in flags if f.color1 == color1 and f.color2 == color2]))
                                    s.add(p)

        s.commit()

        self.colors = colors

        s.close()

    def get_flags(self, color1=None, color2=None):
        s = self.Session()

        if color1 is None:
            flags = s.query(Flag).all()

        elif color2 is None:
            flags = s.query(Flag).filter_by(color1=color1).all()

        else:
            flags = s.query(Flag).filter_by(color1=color1, color2=color2).all()

        s.close()

        return flags

    def get_probabilities(self, req):
        s = self.Session()

        d = dict()

        for c in self.colors:
            d[c] = 0

        if len(req.color1) == 0:
            probs = s.query(Probability).filter_by(color2="").all()

            for p in probs:
                d[p.color1] = p.value

        elif len(req.color2) == 0:
            probs = s.query(Probability).filter_by(color1=req.color1, color3="").all()

            for p in probs:
                d[p.color1] = p.value

        else:
            probs = s.query(Probability).filter_by(color1=req.color1, color2=req.color2).all()

            for p in probs:
                d[p.color1] = p.value

        s.close()

        colors = []
        probs = []

        for k, v in d.items():
            colors.append(k)
            probs.append(v)

        return GetProbabilitiesResponse(colors=colors, probabilities=probs)


def main():
    # ---------------------------------------------------
    # INITIALIZATION
    # ---------------------------------------------------
    default_node_name = 'database'
    rospy.init_node(default_node_name, anonymous=False)

    database = Database()

    rospy.spin()


if __name__ == "__main__":
    main()