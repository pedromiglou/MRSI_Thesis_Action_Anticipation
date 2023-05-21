#!/usr/bin/env python3

# pip install sqlalchemy
# pip install psycopg2

import rospy
import yaml

from itertools import permutations
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from base_controller import BaseController
from db_models import Base, Flag, Probability


class ProbabilitiesController(BaseController):
    """
    Class containing methods to communicate with the database.
    """

    def __init__(self, position_list, flags_path):
        # Scheme: "postgres+psycopg2://<USERNAME>:<PASSWORD>@<IP_ADDRESS>:<PORT>/<DATABASE_NAME>"
        self.engine = create_engine('postgresql+psycopg2://postgres:password@localhost:5432/postgres')

        self.Session = sessionmaker(bind=self.engine)

        self.recreate_db(flags_path)

        super().__init__(position_list)

    def recreate_db(self, flags_path):
        # recreate all tables
        Base.metadata.drop_all(self.engine)
        Base.metadata.create_all(self.engine)

        s = self.Session()

        # add flags to database
        for data in yaml.load_all(open(flags_path)):
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

        for p in permutations(colors, 1):
            if len(flags) != 0:
                s.add(Probability(color1=p[0], color2="", color3="", value=len([f for f in flags if f.color1 == p[0]])/len(flags)))
        
        for p in permutations(colors, 2):
            if len([f for f in flags if f.color1 == p[0]]) != 0:
                s.add(Probability(color1=p[0], color2=p[1], color3="",
                            value=len([f for f in flags if f.color1 == p[0] and f.color2 == p[1]]) / len([f for f in flags if f.color1 == p[0]])))
        
        for p in permutations(colors, 3):
            if len([f for f in flags if f.color1 == p[0] and f.color2 == p[1]]) != 0:
                s.add(Probability(color1=p[0], color2=p[1], color3=p[2],
                            value=len([f for f in flags if f.color1 == p[0] and f.color2 == p[1] and f.color3 == p[2]]) /
                                                    len([f for f in flags if f.color1 == p[0] and f.color2 == p[1]])))

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

    def get_probabilities(self, color1="", color2=""):
        s = self.Session()

        d = dict()

        for c in self.colors:
            d[c] = 0

        if len(color1) == 0:
            probs = s.query(Probability).filter_by(color2="").all()

            for p in probs:
                d[p.color1] = p.value

        elif len(color2) == 0:
            probs = s.query(Probability).filter_by(color1=color1, color3="").all()

            for p in probs:
                if p.color2 != "":
                    d[p.color2] = p.value

        else:
            probs = s.query(Probability).filter_by(color1=color1, color2=color2).all()

            for p in probs:
                if p.color3 != "":
                    d[p.color3] = p.value

        s.close()

        return d.items()

    def idle_state(self):
        if self.current_block is not None: # get colors from database, else wait for user input
            if len(self.blocks) % 3 == 0:
                return
            elif len(self.blocks) % 3 == 1:
                probs = self.get_probabilities(color1=self.blocks[-1])
            elif len(self.blocks) % 3 == 2:
                probs = self.get_probabilities(color1=self.blocks[-2], color2=self.blocks[-1])

            probs = sorted(probs, key=lambda x: x[1], reverse=True)

            self.current_block = [p[0] for p in probs if p[1] > 0]

            if self.state == "idle":
                if len(self.current_block)>0:
                    self.state = "picking_up"
                else:
                    self.current_block = None


def main():
    # ---------------------------------------------------
    # INITIALIZATION
    # ---------------------------------------------------
    default_node_name = 'probabilities_controller'
    rospy.init_node(default_node_name, anonymous=False)

    quaternion_poses = rospy.get_param(rospy.search_param('quaternion_poses'))
    flags_path = rospy.get_param(rospy.search_param('flags_path'))

    probabilities_controller = ProbabilitiesController(position_list = quaternion_poses, flags_path=flags_path)

    probabilities_controller.loop()

    rospy.spin()

    probabilities_controller.arm_gripper_comm.gripper_disconnect()


if __name__ == "__main__":
    main()
