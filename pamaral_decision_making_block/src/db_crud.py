#!/usr/bin/env python3

# pip install sqlalchemy
# pip install psycopg2

import rospy
import yaml

from pamaral_decision_making_block.db_models import Base, Flag
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

    def recreate_db(self):
        Base.metadata.drop_all(self.engine)
        Base.metadata.create_all(self.engine)

        s = self.Session()

        for data in yaml.load_all(open("/home/miglou/catkin_ws/src/MRSI_Thesis/data/flags.yaml")):
            flag = Flag(**data)
            s.add(flag)

        s.commit()

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
