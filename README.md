# [Recognition of Human Grasping Patterns for Intention Prediction in Collaborative Tasks](https://github.com/pedromiglou/MRSI_Thesis_Action_Anticipation)

## Important Links

- [Sensors Article](https://www.mdpi.com/1424-8220/23/21/8989)
- [Kaggle Dataset](https://www.kaggle.com/datasets/pedromiglou/human-grasping-patterns-for-object-recognition)

## Installation Guide

The code in this repository was made to work with:
- Ubuntu 20.04.3 LTS
- ROS Noetic
- UR10e manipulator (Universal Robot 10 e-series)
- 2 Astra Pro RGB-D cameras

1. Install [ROS Noetic](https://wiki.ros.org/noetic/Installation/Ubuntu).

2. (Optional) Install [Docker](https://docs.docker.com/engine/install/ubuntu/) to train machine learning models inside a container or to run the probability-based decision making node.

3. Follow the instructions in the following repositories to install them:
    - [larcc_drivers](https://github.com/lardemua/larcc_drivers)
    - [object_grasping_pattern_recognition](https://github.com/lardemua/object_grasping_pattern_recognition)
    - [ros_astra_camera package](https://github.com/orbbec/ros_astra_camera)

4. Clone this repository and [usb_cam](https://github.com/ros-drivers/usb_cam) into your catkin workspace:

    ```
    cd ~/catkin_ws/src
    git clone https://github.com/pedromiglou/MRSI_Thesis_Action_Anticipation.git
    git clone https://github.com/ros-drivers/usb_cam.git
    ```

5. Install additional system dependencies:

    ```
    sudo apt install libpq-dev libv4l-dev python3-pip v4l-utils
    ```

6. Compile the catkin workspace:

    ```
    cd ~/catkin_ws && catkin_make
    ```

6. Install python requirements:

    ```
    cd ~/catkin_ws/src/MRSI_Thesis_Action_Anticipation
    pip install -r requirements.txt
    ```

## Usage Guide

1. Launch the system:

    ```
    roslaunch pamaral_bringup pamaral_bringup_all.launch
    ```

2. Launch one of the decision making node:

    ```
    roslaunch pamaral_decision_making_block base_controller.launch
    ```
    
    or

    ```
    roslaunch pamaral_decision_making_block probabilities_controller.launch
    ```

    or

    ```
    roslaunch pamaral_decision_making_block rule_based_controller.launch
    ```

    or

    ```
    roslaunch pamaral_decision_making_block rule_based_controller+mediapipe.launch
    ```
