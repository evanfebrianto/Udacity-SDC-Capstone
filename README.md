This is the project repo for the final project of the Udacity Self-Driving Car Nanodegree: Programming a Real Self-Driving Car. For more information about the project, see the project introduction [here](https://classroom.udacity.com/nanodegrees/nd013/parts/6047fe34-d93c-4f50-8336-b70ef10cb4b2/modules/e1a23b06-329a-4684-a717-ad476f0d8dff/lessons/462c933d-9f24-42d3-8bdc-a08a5fc866e4/concepts/5ab4b122-83e6-436d-850f-9f4d26627fd9).

Please use **one** of the two installation options, either native **or** docker installation.

### Native Installation

* Be sure that your workstation is running Ubuntu 16.04 Xenial Xerus or Ubuntu 14.04 Trusty Tahir. [Ubuntu downloads can be found here](https://www.ubuntu.com/download/desktop).
* If using a Virtual Machine to install Ubuntu, use the following configuration as minimum:
  * 2 CPU
  * 2 GB system memory
  * 25 GB of free hard drive space

  The Udacity provided virtual machine has ROS and Dataspeed DBW already installed, so you can skip the next two steps if you are using this.

* Follow these instructions to install ROS
  * [ROS Kinetic](http://wiki.ros.org/kinetic/Installation/Ubuntu) if you have Ubuntu 16.04.
  * [ROS Indigo](http://wiki.ros.org/indigo/Installation/Ubuntu) if you have Ubuntu 14.04.
* Download the [Udacity Simulator](https://github.com/udacity/CarND-Capstone/releases).

### Docker Installation
[Install Docker](https://docs.docker.com/engine/installation/)

Build the docker container
```bash
docker build . -t capstone
```

Run the docker file
```bash
docker run -p 4567:4567 -v $PWD:/capstone -v /tmp/log:/root/.ros/ --rm -it capstone
```

### Port Forwarding
To set up port forwarding, please refer to the "uWebSocketIO Starter Guide" found in the classroom (see Extended Kalman Filter Project lesson).

### Usage

1. Clone the project repository
```bash
git clone https://github.com/udacity/CarND-Capstone.git
```

2. Install python dependencies
```bash
cd CarND-Capstone
pip install -r requirements.txt
```
3. Make and run styx
```bash
cd ros
catkin_make
source devel/setup.sh
roslaunch launch/styx.launch
```
4. Run the simulator

### Real world testing
1. Download [training bag](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/traffic_light_bag_file.zip) that was recorded on the Udacity self-driving car.
2. Unzip the file
```bash
unzip traffic_light_bag_file.zip
```
3. Play the bag file
```bash
rosbag play -l traffic_light_bag_file/traffic_light_training.bag
```
4. Launch your project in site mode
```bash
cd CarND-Capstone/ros
roslaunch launch/site.launch
```
5. Confirm that traffic light detection works on real life images

### Other library/driver information
Outside of `requirements.txt`, here is information on other driver/library versions used in the simulator and Carla:

Specific to these libraries, the simulator grader and Carla use the following:

|        | Simulator | Carla  |
| :-----------: |:-------------:| :-----:|
| Nvidia driver | 384.130 | 384.130 |
| CUDA | 8.0.61 | 8.0.61 |
| cuDNN | 6.0.21 | 6.0.21 |
| TensorRT | N/A | N/A |
| OpenCV | 3.2.0-dev | 2.4.8 |
| OpenMP | N/A | N/A |

We are working on a fix to line up the OpenCV versions between the two.

## Project Overview
Carla is a self-driving vehicle built by Udacity from a customized Lincoln MKZ. Its self-driving system is divided into four primary components: **Sensors**, **perception**, **planning**, and **control** are all important parts of the process.

![carla_architecture](imgs/carla_architecture.png)

### Sensors
Includes everything needed to understand its surroundings and location including **cameras**, **lidar**, **GPS**, **radar**, and **IMU**
#### Perception
Abstracts sensor inputs into object **detection** and **localization**
##### Detection
* Vehicle detection, traffic light detection, obstacle identification, and other software pipelines are included.
* Image processing techniques include feature extraction using the Histogram of Oriented Gradients (HOG), color transforms, and spatial binning.
* For repeated detections, sliding-window or sub-sampling, as well as heat maps and bounding boxes, are used as classification methods.
##### Localization
* Answers the question: “Where is our car in a given map with an accuracy of 10cm or less?”
* Based on the notion that GPS is not accurate enough
* Onboard sensors are used to estimate transformation between measurements and a given map
#### Planning
Path planning is broken down into for sub-components: **route planning**, **prediction**, **behavioral planning**, and **trajectory planning**
##### Route Planning
The route planning component is in charge of making high-level decisions about how to get a vehicle from one place on a map to another, such as which roads, highways, or freeways to use. This capability is similar to route planning on many smartphones and new car navigation systems.
##### Prediction
The prediction component forecasts future actions that other objects may take. If another vehicle is detected, for example, the prediction component will calculate its future course.
##### Behavioral Planning
The behavioral planning component determines the vehicle's behavior at any given point in time. This component may issue maneuvers such as stopping at a traffic light or intersection, changing lanes, accelerating, or making a left turn onto a new street.
##### Trajectory Planning
The trajectory planning component will identify which trajectory is appropriate for executing the desired immediate behavior based on the desired immediate behavior.
### Control
The control component takes the trajectory outputs and adjusts the control inputs for smooth vehicle operation using a controller algorithm like **PID** or **MPC**.

### ROS Architecture

Different nodes (written in Python or C++) connect with each other via ROS messages in the ROS Architecture. The nodes are illustrated in the diagram below, along with their communication with one another. The ROS nodes are represented as ovally outlined text boxes inside rectangular boxes, while the topics that are subscribed or published to are represented by simple rectangular boxes. The flow of communication is shown by the direction of the arrows.

![rosgraph](imgs/rosgraph.png)

The styx server, which connects the simulator and ROS by delivering information about the car's condition and surroundings (car's current position, velocity, and images from the front camera) and accepting control input, is the most central node in the rqt-graph (steering, braking, throttle). The three central tasks of perception, planning, and control are represented by the other nodes.

To recognize traffic lights, the photos are analyzed by a trained neural network within the traffic light classifier. The traffic light detector receives the percepted state of a potentially approaching traffic light, as well as the car's current pose and a set of base waypoints from the waypoint loader. The traffic light detector can publish a waypoint near the next traffic light where the automobile should stop if the light is red using this often incoming data.

The waypoint updater node can plan acceleration / deceleration and publish it to the waypoint follower node using the traffic light detector's subscribed information and subscriptions to base waypoints. This node communicates with the DBW (Drive by Wire) node, which handles the duty of autonomously directing the car. It also accepts the current velocity of the automobile (straight from the car / simulator) as an input and outputs steering, braking, and throttle commands.

### Node Design

![archi](imgs/architecture.png)

The node design of those nodes that are built within this project will be discussed in this paragraph. Waypoint updater (waypoint updater.py), traffic light detector (tl detector.py), and drive by wire node (dbw node.py) are the three modules.

#### Waypoint Updater
Because it specifies which waypoints the car should follow, the waypoint updater node plays a crucial role in the planning work. The node is divided into several sections: First, there's an import section, which imports certain Python libraries and message formats. The initialization of various constants that are not intended to be altered follows. The class WaypointUpdater is introduced after this section. The WaypointUpdater is divided into several sections. The class's attributes are defined by the init-function, which also determines which topics the class subscribes to and which it publishes on.

The following routines are either generic methods or callback functions that the subscribers in the init-function invoke frequently. The basic waypoints (output of waypoint loader), the car's posture (simulator / car), and the traffic waypoint (output of tl detector) are all frequently called. The decelerate waypoints-function, which incorporates a square-root shaped deceleration towards a predefined stopline location in the case of red traffic lights, is the most essential general technique. The main function at the end of the node runs the node and logs an error if ROS is halted for any reason.

#### Traffic Light Detection
The traffic light detector has the same structure as the Waypoint Updater, with an import or startup section followed by a class containing properties and functions. Finally, the TL detection subroutine compiles the code using its main function. The subscriptions to the current location base waypoints, the specified traffic light array containing the ground-truth coordinates of the traffic lights, and the identified color of the traffic light are all included in the TLDetector class's init-function.

The color of the traffic light is determined by the traffic light classifier, which is a neural network described in greater detail in the following paragraph. The callback image_cb updates the topic image color, which in turn calls the method process traffic lights(), which in turn calls the function get light state(), which receives the traffic light classification. This subroutine eventually publishes the waypoint to halt at for each upcoming detected red traffic signal.

#### Drive-By-Wire (DBW) Node
The dbw node, which is responsible for driving the car, is the third node that I wrote. It uses a twist controller, which uses a PID-controller and a Lowpass filter to output throttle, brake, and steering values. If dbw_enabled is set to true, the dbw node directly publishes throttle, brake, and steering commands for the car/simulator.

### Neural Network Design

#### Model
The traffic light classification model is based on the pre-trained on the COCO dataset model "faster_rcnn_resnet101_coco" from [Tensorflow detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md). Using the [Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection), the simulator data model and real data model I trained. 

The models are available in the `ros/src/tl_detector/light_classification/train_model` directory. 

#### Dataset
Step-by-step [Tensorflow Object Detection API tutorial](https://medium.com/@WuStangDan/step-by-step-tensorflow-object-detection-api-tutorial-part-1-selecting-a-model-a02b6aabe39e) was a good guide of using the Tensorflow object detection API for traffic light classification. 

The simulator dataset was from [here](https://drive.google.com/file/d/0Bw5abyXVejvMci03bFRueWVXX1U), and the real dataset was from [here](https://drive.google.com/file/d/0B-Eiyn-CUQtxdUZWMkFfQzdObUE).

#### Classification
The classification output has four categories: Red, Green, Yellow and off. To simplify, the final output will be Red or Non-Red, that is only the Red will be classified as `TrafficLight.RED`, and the other cases will be classified as `TrafficLight.GREEN`.

#### Examples of Simulator Testing Results:

![sim_red](https://user-images.githubusercontent.com/12635332/39411764-80e4135a-4bc5-11e8-90de-be830ed9ffcb.png)
![sim_green](https://user-images.githubusercontent.com/12635332/39411770-96841c50-4bc5-11e8-8ffb-bd41fb549881.png)
![sim-yellow](https://user-images.githubusercontent.com/12635332/39411774-9f08b6c4-4bc5-11e8-8921-3fefcad68e04.png)
![sim_none](https://user-images.githubusercontent.com/12635332/39411776-a7a32d78-4bc5-11e8-8034-bbc0066d8b30.png)

#### Examples of Real Testing Results:

![real_red](https://user-images.githubusercontent.com/12635332/39411790-e24bf022-4bc5-11e8-95b2-55a7fd07ddf5.png)
![real_green](https://user-images.githubusercontent.com/12635332/39411788-d874bbb0-4bc5-11e8-866f-1496f7f47596.png)
![real_yellow](https://user-images.githubusercontent.com/12635332/39412245-4e0ad37a-4bce-11e8-9312-7f727d085676.png)
![real_none](https://user-images.githubusercontent.com/12635332/39412259-8b125892-4bce-11e8-9e59-64a689a7eb99.png)

## Results

Following the completion of the program, the walkthrough videos were uploaded to the Udacity project page. There was a rewrite attempt to conform to the walkthrough technique because they had a much more elegant approach to implementing the nodes.

![result](imgs/capstone_run.gif)

Setting up the environment was one of the most difficult aspects of this project. I misconfigured the Python environment in the VM and was stranded for several hours. After going to the knowledge forum, one of the mentors assisted me greatly in resolving this issue.

In order to accomplish the project, I was also perplexed by how the car should interact and operate within the test lot. For me, the test track is simple to execute because there are basic rules of the road that must be observed in order to effectively travel the track. There are also standardized movements that can be performed on the road that lead to simple vehicle behavior trajectories on the test track. The test lot, on the other hand, is troublesome and peculiar due to the lack of defined road rules and maneuvers to conduct within the test lot.

Another thing I discovered is that while docker setup is much easier, the latency is significantly higher than VM or native setup. As a result, I used VM to run ROS nodes and my native Ubuntu 20.04 LTS to run the simulator for this project.

Overall, this project was both tough and gratifying.