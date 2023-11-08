# Recognition of Human Grasping Patterns for Intention Prediction in Collaborative Tasks

MRSI Thesis Repository

<!-- ## Repository Structure

- data: ROS package containing datasets and other data
- dissertation_reports: pdf reports and latex files
- model_training: model training scripts, Dockerfile and results
- pamaral_bringup: system launch files
- pamaral_decision_making_block: decision making nodes
- pamaral_models: 
-->

## Important Links

- [Dataset published on Kaggle](https://www.kaggle.com/datasets/pedromiglou/human-grasping-patterns-for-object-recognition)

## Installation Guide (In Progress)

The code in this repository was made to work with:
- Ubuntu 20.04.3 LTS
- ROS Noetic
- UR10e manipulator (Universal Robot 10 e-series)
- 2 Astra Pro RGB-D cameras

1. Install [ROS Noetic](https://wiki.ros.org/noetic/Installation/Ubuntu).

2. (Optional) Install [Docker](https://docs.docker.com/engine/install/ubuntu/) to train machine learning models inside a container or to run the probability-based decision making node.

3. Follow the instructions in the following repositories to install them:
    - [larcc_interface](https://github.com/afonsocastro/larcc_interface)
    - [ros_astra_camera package](https://github.com/orbbec/ros_astra_camera)
    
    Change the branch of larcc_interface to `stop-mid-movement`:
    ```
    cd ~/catkin_ws/src/larcc_interface && git checkout stop-mid-movement
    ```

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

## Other thesis in action anticipation

- https://repository.kaust.edu.sa/handle/10754/673882
