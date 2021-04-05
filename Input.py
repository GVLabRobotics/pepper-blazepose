import qi
import pandas as pd
import nep
import sys

msg_type = "json"

node = nep.node("publisher_sample")
pub = node.new_pub("test", msg_type)


df = pd.read_csv('#file.csv')
columns = list(df.columns.values)
JointAngles = [h for h in columns if 'TimeStamp' not in h]

# Command angles
HipRoll = df['HipRoll'].values.tolist()
HipPitch = df['HipPitch'].values.tolist()
HeadYaw = df['HeadYaw'].values.tolist()
HeadPitch = df['HeadPitch'].values.tolist()
LShoulderPitch = df['LShoulderPitch'].values.tolist()
RShoulderPitch = df['RShoulderPitch'].values.tolist()
LElbowYaw = df['LElbowYaw'].values.tolist()
RElbowYaw = df['RElbowYaw'].values.tolist()
LShoulderRoll = df['LShoulderRoll'].values.tolist()
RShoulderRoll = df['RShoulderRoll'].values.tolist()
LElbowRoll = df['LElbowRoll'].values.tolist()
RElbowRoll = df['RElbowRoll'].values.tolist()
TimeStamp = df['TimeStamp'].values.tolist()

movement = {'RShoulderRoll': RShoulderRoll, 'LShoulderRoll': LShoulderRoll,
            'RElbowRoll': RElbowRoll, 'LElbowRoll': LElbowRoll, 'HeadYaw': HeadYaw,
            'HeadPitch': HeadPitch, 'LShoulderPitch': LShoulderPitch,
            'RShoulderPitch': RShoulderPitch, 'LElbowYaw': LElbowYaw,
            'RElbowYaw': RElbowYaw, 'HipRoll': HipRoll, 'HipPitch': HipPitch}


ip = "192.168.11.41"
port = 9559
run = True

session = qi.Session()

try:
    session.connect("tcp://" + ip + ":" + str(port))
except RuntimeError:
    print ("Can't connect to Naoqi at ip \"" + ip + "\" on port " + str(port) +".\n"
           "Please check your script arguments. Run with -h option for help.")
    sys.exit(1)

motion_service = session.service("ALMotion")
motion_service.setStiffnesses("Head", 1.0)
motion_service.setStiffnesses("LArm", 1.0)
motion_service.setStiffnesses("RArm", 1.0)

names  = JointAngles
angleLists = [movement[JointAngles[i]] for i in range(len(JointAngles))]
timeLists = [TimeStamp for j in range(len(JointAngles))]
isAbsolute  = True

pub.publish({"msg": True})

motion_service.angleInterpolation(names, angleLists, timeLists, isAbsolute)

pub.publish({"msg": False})

motion_service.setStiffnesses("Head", 0.0)
motion_service.setStiffnesses("LArm", 0.0)
motion_service.setStiffnesses("RArm", 0.0)
run = False


print("I finished")
