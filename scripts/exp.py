# import rospy
# import numpy as np
# from geometry_msgs.msg import Pose, Point, Quaternion
# from iiwa_cam.msg import GeneralControl
# from iiwa_cam.srv import GeneralExecution, GeneralPlan

###############################

from __future__ import print_function

import csv
import threading
import time
from datetime import datetime

import geometry_msgs.msg
import matplotlib.pyplot as plt
import numpy as np
import rospy
import tf.transformations as tft
from iiwa_cam.msg import GeneralControl
from iiwa_cam.srv import GeneralExecution, EndEffectorState
from iiwa_cam.srv import GeneralPlan
from iiwa_msgs.msg import JointPositionVelocity

##############################
store_folder = "/home/other/Desktop/screwdriver_exp/sensor_single"  ## Fix me

test = False  # Just run test screwdriving, no data capture

nominal_hole = [0.82352, -0.1634, 0.3095, 0.31629, 0.31504, 0.63268, 0.63279]
static_offset = [
    3.914695198,
    -0.3579985943,
    -7.03114873,
    -1.519546672,
    0.4173471108,
    -0.9741524884,
]


def get_traj_exec(request):
    rospy.wait_for_service("general_execution")

    node = rospy.ServiceProxy("general_execution", GeneralExecution)
    result = node(request)

    return result


def get_traj_plan_test(poses, speed=0.05, stiffness=1500):
    # Poses is list of geometry_msgs/poses (waypoint for plan) of tcp (x,y,z,rx,ry,rz,w)
    # Ex: Grab transform of iiwa_orange_gripper_ee
    # From command line: trans = tfBuffer.lookup_transform(turtle_name, 'iiwa_orange_gripper_ee', rospy.Time())

    # Speed is percentage of max joint angle speed

    # rosrun iiwa_cam waypoints_plan_execute.py"general_plan")

    request = GeneralControl(
        "iiwa_orange",
        poses,
        stiffness,
        speed,
        True,
        False,  # last true only if giving start point as current point
    )

    node = rospy.ServiceProxy("general_plan", GeneralPlan)
    result = node(request)

    return result


def uniform_sampling_around_hole(nominal_hole_location):
    pass


def project_backwards_by_distance(nominal_hole_location, distance):
    pass


def safe_getattr(obj, attr_chain, default=None):
    for attr in attr_chain:
        if isinstance(obj, (list, tuple)) and isinstance(attr, int):
            # Handle list or tuple indexing
            try:
                obj = obj[attr]
            except IndexError:
                print(f"Index Error - {attr}")
                return default
        elif isinstance(obj, dict) and (isinstance(attr, str) or isinstance(attr, int)):
            # Handle dictionary key access
            obj = obj.get(attr, default)
            if obj is default:
                return default
        else:
            # Handle object attribute access
            obj = getattr(obj, attr, default)
            if obj is default:
                return default
    return obj


def get_traj_plan(
        robot_name,
        robot_poses,
        robot_stiffness,
        robot_speed,
        wait_execution_finished=True,
        dumy_idx=-1,
        pre_record=False,
):
    rospy.wait_for_service("general_plan")

    node = rospy.ServiceProxy("general_plan", GeneralPlan)

    request = GeneralControl(
        robot_name,
        robot_poses,
        robot_stiffness,
        robot_speed,
        wait_execution_finished,
        False,
    )
    result = node(request)
    return result


class Worker:
    def __init__(self):
        self.running = True
        self._recorded_data = list()
        self.joint_position_velocity = None

        self.joint_position_velocity_subscriber = rospy.Subscriber(
            "/iiwa_orange/state/JointPositionVelocity",
            JointPositionVelocity,
            self.joint_position_velocity_callback,
        )

        self.thread = threading.Thread(target=self.store_data_to_object, daemon=True)

    def joint_position_velocity_callback(self, data):
        self.joint_position_velocity = data

    def store_data_to_object(self):
        rospy.wait_for_service("/cam/iiwa/EndEffectorState")
        state_node = rospy.ServiceProxy("/cam/iiwa/EndEffectorState", EndEffectorState)

        # tf_buffer = tf2_ros.Buffer()
        # tfListener = tf2_ros.TransformListener(tf_buffer)
        # tf_buffer.can_transform("robot_origin/base_link", "bucket_SC/base_link", rospy.Time(0), rospy.Duration(3.0))

        while self.running:
            time.sleep(0.01)

            current_state = state_node("iiwa_orange")
            # transforms = [tf_buffer.lookup_transform("robot_origin/base_link", "bucket_SC/base_link", rospy.Time(0)).transform]
            # joint_position_velocity = rospy.wait_for_message("/iiwa_blue/state/JointPositionVelocity", JointPositionVelocity)

            # Fetch the most recent JointPositionVelocity message from the class attribute updated by the subscriber callback
            joint_position_velocity = self.joint_position_velocity

            self._recorded_data.append(
                [
                    datetime.now(),
                    safe_getattr(current_state, ["pose", "position", "x"]),
                    safe_getattr(current_state, ["pose", "position", "y"]),
                    safe_getattr(current_state, ["pose", "position", "z"]),
                    safe_getattr(current_state, ["pose", "orientation", "x"]),
                    safe_getattr(current_state, ["pose", "orientation", "y"]),
                    safe_getattr(current_state, ["pose", "orientation", "z"]),
                    safe_getattr(current_state, ["pose", "orientation", "w"]),
                    safe_getattr(current_state, ["velocity", "linear", "x"]),
                    safe_getattr(current_state, ["velocity", "linear", "y"]),
                    safe_getattr(current_state, ["velocity", "linear", "z"]),
                    safe_getattr(current_state, ["velocity", "angular", "x"]),
                    safe_getattr(current_state, ["velocity", "angular", "y"]),
                    safe_getattr(current_state, ["velocity", "angular", "z"]),
                    safe_getattr(current_state, ["wrenches", -1, "force", "x"]),
                    safe_getattr(current_state, ["wrenches", -1, "force", "y"]),
                    safe_getattr(current_state, ["wrenches", -1, "force", "z"]),
                    safe_getattr(current_state, ["wrenches", -1, "torque", "x"]),
                    safe_getattr(current_state, ["wrenches", -1, "torque", "y"]),
                    safe_getattr(current_state, ["wrenches", -1, "torque", "z"])
                    # convert_mocab_points_to_str(transforms) #buffer
                ]
            )

    def set_running(self, val):
        self.running = val

    @property
    def recorded_data(self):
        return self._recorded_data


def write_data_to_file(recorded_data, file_name):
    print(f"writing to the file -> {store_folder}/{file_name}")
    with open(f"{store_folder}/{file_name}", "w+") as file:
        csv_writer = csv.writer(file)

        csv_writer.writerow(
            [
                "time",
                "X",
                "Y",
                "Z",
                "x",
                "y",
                "z",
                "w",
                "Vx",
                "Vy",
                "Vz",
                "Wx",
                "Wy",
                "Wz",
                "fx",
                "fy",
                "fz",
                "tx",
                "ty",
                "tz",
            ]
        )

        for row in recorded_data:
            csv_writer.writerow(row)


def quaternion_rotation_matrix(Q):
    """
    Covert a quaternion into a full three-dimensional rotation matrix.

    Input
    :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3)

    Output
    :return: A 3x3 element matrix representing the full 3D rotation matrix.
             This rotation matrix converts a point in the local reference
             frame to a point in the global reference frame.
    """
    # Extract the values from Q
    q0 = Q[0]
    q1 = Q[1]
    q2 = Q[2]
    q3 = Q[3]

    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)

    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)

    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1

    # 3x3 rotation matrix
    rot_matrix = np.array([[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]])

    return rot_matrix


def projectPose(pose, array):
    # print("Old pose",pose)
    # print("projection array",array)
    x0, y0, z0 = [pose[0], pose[1], pose[2]]
    dx, dy, dz = array
    quat = [pose[3], pose[4], pose[5], pose[6]]
    # r = R.from_quat(quat)
    r = quaternion_rotation_matrix([pose[6], pose[3], pose[4], pose[5]])
    # print("r: ",r)
    # vxx, vxy, vxz #VecX
    # vyx, vyy, vyz #VecY
    # vzx, vzy, vzz #VecZ
    vxx, vyx, vzx = r[0]
    vxy, vyy, vzy = r[1]
    vxz, vyz, vzz = r[2]
    new_point = [
        x0 + dx * vxx + dy * vyx + dz * vzx,
        y0 + dx * vxy + dy * vyy + dz * vzy,
        z0 + dx * vxz + dy * vyz + dz * vzz,
    ]
    new_pose = new_point + quat
    # print("New pose",new_pose)
    return new_pose


def getPose_msg(pose):
    point = geometry_msgs.msg.Point(pose[0], pose[1], pose[2])
    quat = geometry_msgs.msg.Quaternion(pose[3], pose[4], pose[5], pose[6])
    pose_msg = geometry_msgs.msg.Pose(point, quat)
    return pose_msg


def rotate_quaternion_z(quat: geometry_msgs.msg.Quaternion, deg: int):
    rad = np.deg2rad(deg)

    rotation_quat = tft.quaternion_about_axis(rad, (1, 0, 0))
    quat_tuple = (quat.x, quat.y, quat.z, quat.w)

    new_quat_tuple = tft.quaternion_multiply(rotation_quat, quat_tuple)

    new_quat = geometry_msgs.msg.Quaternion(*new_quat_tuple)

    return new_quat


if __name__ == "__main__":
    if test:
        rospy.init_node("waypoints_plan_execute", anonymous=True)
        # new terminal
        # roscore
        # Move key to lock and check in auto mode. Make sure e-stop disengaged.
        # start the ROSSmartServo application on pendant (unless just running sim)
        # Press play on pendant

        # new terminal
        # cd /home/other/iiwa_screwdriving_package/catkin_ws
        # source devel/setup.zsh
        # roslaunch iiwa_cam moveit_kuka_pipeline.launch real_robot_execution:=false
        # *(true if running robot)

        # cd /home/other/iiwa_screwdriving_package
        # ./install.sh
        # *(moves files into workspace that is gitignored)
        # cd catkin_ws
        # source devel/setup.zsh
        #  cd .. && ./install.sh && cd catkin_ws && source devel/setup.zsh
        # rosrun iiwa_cam waypoints_plan_execute.py

        # To get live waypoint: run with real_robot_execution:=true
        # In RViz. Add-TF. Unselect all. Add iiwa_orange_gripper_ee

        # poses = generate_poses(waypoints_go_back)
        # poses = generate_poses(waypoints)

        # Update nominal hole location as needed (if moving robot or using new hole)
        # poses = generate_poses(nominal_waypoint)

        # test_hole_location=uniform_sampling_around_hole(nominal_hole_location,trial_num)
        # test_plunge_location=project_backwards_by_distance(nominal_hole_location,trial_num)

        orange_hover = getPose_msg(projectPose(nominal_hole, [0, 0.07, 0]))
        orange_screw_hole = getPose_msg(nominal_hole)
        orange_screw_plunge = getPose_msg(projectPose(nominal_hole, [0, -0.01, 0]))

        res = get_traj_plan_test([orange_hover, orange_screw_hole, orange_screw_plunge])
        while res.success:
            replan = input("replan? [y/n]")
            if replan == "y":
                res = get_traj_plan_test([orange_hover])
            elif replan == "n" or replan == "":
                break

        if res.success:
            execute = input("execute? [y/n]")
            if execute == "y":
                get_traj_exec(res.general_traj)
    else:
        ring_num = 2
        trial_screw = "m4"

        offset_r = 0.001 * ring_num
        offset_deg = max(15 * (ring_num - 1), 0)
        offset_orientation = 15

        offset_z = -offset_r * np.cos(np.deg2rad(offset_deg))
        offset_x = offset_r * np.sin(np.deg2rad(offset_deg))

        orange_hover = getPose_msg(
            projectPose(nominal_hole, [offset_x, 0.07, offset_z])
        )
        orange_screw_hole = getPose_msg(
            projectPose(nominal_hole, [offset_x, 0, offset_z])
        )
        orange_screw_plunge = getPose_msg(
            projectPose(nominal_hole, [offset_x, -0.025, offset_z])
        )

        orange_hover.orientation = rotate_quaternion_z(
            orange_hover.orientation, offset_orientation
        )
        orange_screw_hole.orientation = rotate_quaternion_z(
            orange_screw_hole.orientation, offset_orientation
        )
        orange_screw_plunge.orientation = rotate_quaternion_z(
            orange_screw_plunge.orientation, offset_orientation
        )

        response = get_traj_plan_test([orange_hover], 0.1, 2000)
        get_traj_exec(response.general_traj)

        # response = get_traj_plan_test([orange_screw_hole], 0.1, 1500)

        data_recorder = Worker()
        data_recorder.thread.start()
        start_time = time.time_ns()

        # get_traj_exec(response.general_traj)

        response = get_traj_plan_test([orange_screw_plunge], 0.05, 1500)
        get_traj_exec(response.general_traj)

        success = input("Success?[1: success, 2[ab]: failure]\n")
        end_time = time.time_ns()

        data_recorder.running = False
        data_recorder.thread.join()

        data_store_file = (
                trial_screw
                + "_"
                + str(ring_num)
                + "_"
                + str(offset_deg)
                + "_"
                + str(offset_orientation)
                + "_"
                + success
        )
        if success == "1":
            data_store_file += "_" + str(round((end_time - start_time) / 10 ** 9, 3))

        data_store_file += ".csv"
        write_data_to_file(data_recorder.recorded_data, data_store_file)

        data = np.array(data_recorder.recorded_data)
        fig, (ax1, ax2) = plt.subplots(2, 1)
        ax1.plot(data[:, 0], data[:, -6:-3])
        ax2.plot(data[:, 0], data[:, -3])
        ax2.plot(data[:, 0], data[:, -2])
        ax2.plot(data[:, 0], data[:, -1])
        plt.show()

        response = get_traj_plan_test([orange_hover], 0.1, 2000)
        get_traj_exec(response.general_traj)

# INSTRUCTIONS TO RUN THE SCRIPT
# 1. Open 5 terminal windows. In each of the window, run `source devel/setup.bash` file
# 2. In one window run `roscore` and always keep it running.
# 3. On the KUKA pendant, switch to auto mode, and select the ROS Smart Servo application and press the play button.
# 4. On one of the terminal windows, run `roslaunch iiwa_cam moveit_kuka_pipeline.launch real_robot_execution:=false`
#   set the flag to true to execute on the real robot.
# 5. On another window, run `rosrun camera_node camera_capture`
# 6. On another window, run `rosrun rosserial_python serial_node.py _port:=/dev/ttyACM0`
# 7. Finally, run `rosrun iiwa_cam waypoints_plan_execute.py`

