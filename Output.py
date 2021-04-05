import pandas as pd
import sys
from naoqi import ALProxy
import time
import nep

df = pd.read_csv('file.csv')
columns = list(df.columns.values)
JointAngles = [h for h in columns if 'TimeStamp' not in h]
TimeStamp = df['TimeStamp'].values.tolist()

# Sensor angles
Hip_Pitch = []
Hip_Roll = []
Head_Pitch = []
Head_Yaw = []
LShoulder_Pitch = []
RShoulder_Pitch = []
LShoulder_Roll = []
RShoulder_Roll = []
LElbow_Roll = []
RElbow_Roll = []
LElbow_Yaw = []
RElbow_Yaw = []

sensor = {'RShoulderRoll': RShoulder_Roll, 'LShoulderRoll': LShoulder_Roll,
          'RElbowRoll': RElbow_Roll, 'LElbowRoll': LElbow_Roll, 'HeadYaw': Head_Yaw,
          'HeadPitch': Head_Pitch, 'LShoulderPitch': LShoulder_Pitch,
          'RShoulderPitch': RShoulder_Pitch, 'LElbowYaw': LElbow_Yaw,
          'RElbowYaw': RElbow_Yaw, 'HipRoll': Hip_Roll, 'HipPitch': Hip_Pitch}

msg_type = "json"
node = nep.node("subscriber_sample")
sub = node.new_sub("test", msg_type)

def main(robotIP):
    PORT = 9559
    run = True
    record = False
    timestamp = TimeStamp[0]/2.0
    print(timestamp)
    
    while run:

        s, msg = sub.listen()

        if s:                   

            if (msg["msg"] == True):
                record = True
                print("start recording")

            if (msg["msg"] == False):
                run = False
                print("finish recording")

        if record:
            try:
                motionProxy = ALProxy("ALMotion", robotIP, PORT)
            except Exception,e:
                print "Could not create proxy to ALMotion"
                print "Error was: ",e
                sys.exit(1)

            useSensors  = True
            for theta in JointAngles:
                q = motionProxy.getAngles(theta, useSensors)
                sensor[theta].append(q[0])

        time.sleep(timestamp)

if __name__ == "__main__":
    robotIp = "192.168.11.41"

    if len(sys.argv) <= 1:
        print "Usage python almotion_getangles.py robotIP (optional default: 127.0.0.1)"
    else:
        robotIp = sys.argv[1]

    main(robotIp)

ThetaSensor = pd.DataFrame.from_dict(sensor)
ThetaSensor.to_csv('body_language_sensor.csv', index=False)
