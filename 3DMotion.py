import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import math
from datetime import datetime

# MediaPipe Pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Input the video
pose = mp_pose.Pose(
    min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = cv2.VideoCapture('#video') # 0 for web camera input

TimeStamp = [] # Time stamps list in real-time

# Define (x, y) coordinates lists
X1 = []  # Left Shoulder X
X2 = []  # Right Shoulder X
X3 = []  # Left Elbow X
X4 = []  # Right Elbow X
X5 = []  # Left Wrist X
X6 = []  # Right Wrist X
X7 = []  # Left Hip X
X8 = []  # Right Hip X
X9 = []  # Nose X

Y1 = []  # Left Shoulder Y
Y2 = []  # Right Shoulder Y
Y3 = []  # Left Elbow Y
Y4 = []  # Right Elbow Y
Y5 = []  # Left Wrist Y
Y6 = []  # Right Wrist Y
Y7 = []  # Left Hip Y
Y8 = []  # Right Hip Y
Y9 = []  # Nose Y

# Loop for each frame
while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    # Convert the BGR image to RGB.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    results = pose.process(image)

    # Draw the pose annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(
        image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    cv2.imshow('MediaPipe Pose', image)
    if cv2.waitKey(5) & 0xFF == 27:
        break

    # Get the current time
    time = datetime.now()
    TimeStamp.append(time)

    # Get the pose coordinates of each landmark
    image_height, image_width, _ = image.shape # Normalize the reference frame according to the resolution of the video.

    # Get the left shoulder coordinates
    X11 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x * image_width
    Y11 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * image_height

    X1.append(X11)
    Y1.append(Y11)

    # Get the right shoulder coordinates
    X12 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * image_width
    Y12 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * image_height

    X2.append(X12)
    Y2.append(Y12)

    # Get the left elbow coordinates
    X13 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].x * image_width
    Y13 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].y * image_height

    X3.append(X13)
    Y3.append(Y13)

    # Get the right elbow coordinates
    X14 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].x * image_width
    Y14 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].y * image_height

    X4.append(X14)
    Y4.append(Y14)

    # Get the left wrist coordinates
    X15 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x * image_width
    Y15 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y * image_height

    X5.append(X15)
    Y5.append(Y15)

    # Get the right wrist coordinates
    X16 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].x * image_width
    Y16 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y * image_height

    X6.append(X16)
    Y6.append(Y16)

    # Get the left hip coordinates
    X23 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x * image_width
    Y23 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y * image_height

    X7.append(X23)
    Y7.append(Y23)

    # Get the right hip coordinates
    X24 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].x * image_width
    Y24 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].y * image_height

    X8.append(X24)
    Y8.append(Y24)

    # Get the nose coordinates
    X0 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * image_width
    Y0 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * image_height

    X9.append(X0)
    Y9.append(Y0)

# Close MediaPipe Pose
pose.close()
cap.release()


# Joint Angles in 2D
xy = {'LeftShoulder': np.column_stack([X1, Y1]), 'RightShoulder': np.column_stack([X2, Y2]),
      'LeftElbow': np.column_stack([X3, Y3]), 'RightElbow': np.column_stack([X4, Y4]),
      'LeftWrist': np.column_stack([X5, Y5]), 'RightWrist': np.column_stack([X6, Y6]),
      'LeftHip': np.column_stack([X7, Y7]), 'RightHip': np.column_stack([X8, Y8])}

# Arms landmarks
landmark1 = 'LeftShoulder'
landmark2 = 'LeftHip'
landmark3 = 'LeftElbow'
landmark4 = 'LeftWrist'
landmark5 = 'RightShoulder'
landmark6 = 'RightHip'
landmark7 = 'RightElbow'
landmark8 = 'RightWrist'

# Calculate each body segment vector in 2D
LS_LE = xy[landmark3] - xy[landmark1]  # Left Shoulder to Left Elbow
LE_LS = xy[landmark1] - xy[landmark3]  # Left Elbow to Left Shoulder
LW_LE = xy[landmark3] - xy[landmark4]  # Left Wrist to Left Elbow
LS_LH = xy[landmark2] - xy[landmark1]  # Left Shoulder to Left Hip

RS_RE = xy[landmark7] - xy[landmark5]  # Right Shoulder to Right Elbow
RE_RS = xy[landmark5] - xy[landmark7]  # Right Elbow to Right Shoulder
RW_RE = xy[landmark7] - xy[landmark8]  # Right Wrist to Right Elbow
RS_RH = xy[landmark6] - xy[landmark5]  # Right Shoulder to Right Hip

RS_LS = xy[landmark1] - xy[landmark5]  # Right Shoulder to Left Shoulder

# Original Lengths of arm segments
l1_left = np.linalg.norm(LE_LS[0, :])  # Upper left arm
l2_left = np.linalg.norm(LW_LE[0, :])  # Lower left arm

l1_right = np.linalg.norm(RE_RS[0, :])  # Upper right arm
l2_right = np.linalg.norm(RW_RE[0, :])  # Lower right arm

torso = Y8[0] - Y2[0]  # Torso original height


# Using Taylor's algorithm to calculate the depth between two orthographic projected points
# Calculate depth between left shoulder and left elbow
Z3 = []
for i in range(LE_LS.shape[0]):
    dz = (l1_left ** 2) - (((X1[i] - X3[i]) ** 2) + ((Y1[i] - Y3[i]) ** 2))
    if dz <= 0.0:
        dz = 0.0
    dz1_left = math.sqrt(dz)
    Z3.append(dz1_left)

# Calculate depth between right shoulder and right elbow
Z4 = []
for i in range(RE_RS.shape[0]):
    dz = (l1_right ** 2) - (((X2[i] - X4[i]) ** 2) + ((Y2[i] - Y4[i]) ** 2))
    if dz <= 0.0:
        dz = 0.0
    dz1_right = math.sqrt(dz)
    Z4.append(dz1_right)

# Calculate depth between left elbow and left wrist
Z5 = []
for i in range(LW_LE.shape[0]):
    dz = (l2_left ** 2) - (((X3[i] - X5[i]) ** 2) + ((Y3[i] - Y5[i]) ** 2))
    if dz <= 0.0:
        dz = 0.0
    dz2_left = math.sqrt(dz)
    end_effector = dz2_left + Z3[i]
    Z5.append(end_effector)

# Calculate depth between right elbow and right wrist
Z6 = []
for i in range(RW_RE.shape[0]):
    dz = (l2_right ** 2) - (((X4[i] - X6[i]) ** 2) + ((Y4[i] - Y6[i]) ** 2))
    if dz <= 0.0:
        dz = 0.0
    dz2_right = math.sqrt(dz)
    end_effector = dz2_right + Z4[i]
    Z6.append(end_effector)

# Robot/Human Angles
LShoulderPitch = []
RShoulderPitch = []
LElbowYaw = []
RElbowYaw = []
RShoulderRoll = []
LShoulderRoll = []
RElbowRoll = []
LElbowRoll = []
HeadYaw = []
HeadPitch = []
HipRoll = []
HipPitch = []

# Calculate the hip roll angles
for i in range(len(X1)):
    # Current horizontal distance between the 2 shoulders over the distance between them
    adj = (X1[i] - X2[i]) / np.linalg.norm(RS_LS[i, :])
    if adj >= 1:          # Keeping the ratio less than or equal to 1
        adj = 1
    phi = np.arccos(adj)  # Arc cos to get the angle
    if phi >= 0.5149:     # Maximum right hip roll is 29.5°.
        phi = 0.5149
    if Y2[i] < Y1[i]:     # If right shoulder is above the left shoulder then the direction of hip roll is reversed.
        phi = phi * -1
    if phi <= -0.5149:    # Maximum left hip roll is -29.5°.
        phi = -0.5149
    HipRoll.append(phi)

# Calculate the hip pitch angles
for i in range(len(Y2)):
    adj = (Y8[i] - Y2[i]) / torso  # Current height of torso over the original height
    if adj >= 1:                   # Keeping the ratio less than or equal to 1
        adj = 1
    phi = np.arccos(adj)           # Hip pitch angle for leaning forward is negative
    if phi >= 1.0385:              # Maximum hip pitch angle is 59.5°.
        phi = 1.0385
    HipPitch.append(phi)

# Calculate the head yaw angles
d = np.linalg.norm(RS_LS[0, :]) / 2  # Half of initial distance between right and left shoulder

for i in range(len(X9)):
    if (X9[i] - X2[i]) / d >= 0.9 and (X9[i] - X2[i]) / d <= 1.1:  # Estimating the angle to be 0° if the nose
        hy = 0.0                                                   # X coordinate doesn't exceed 10% from each side

    elif (X9[i] - X2[i]) / d < 0.9:                                # Angle of looking to the right based on how much
        hy = ((d - (X9[i] - X2[i])) / d) * -(np.pi / 2)            # the nose is approaching the right shoulder.
        if hy <= -np.pi / 2:                                       # Maximum head yaw angle to the right is -90°.
            hy = -np.pi / 2

    elif (X9[i] - X2[i]) / d > 1.1:                                # Angle of looking to the right based on how much
        hy = (((X9[i] - X2[i]) - d) / d) * (np.pi / 2)             # the nose is approaching the left shoulder.
        if hy >= np.pi / 2:                                        # Maximum head yaw angle to the left is 90°.
            hy = np.pi / 2
    HeadYaw.append(hy)

# Calculate the head pitch angles
h = Y2[0] - Y9[0]
for i in range(len(Y9)):
    if (Y2[i] - Y9[i]) / h >= 0.95 and (Y2[i] - Y9[i]) / h <= 1.0:
        hp = 0.0
    elif (Y2[i] - Y9[i]) / h < 0.95:
        hp = ((h - (Y2[i] - Y9[i])) / h) * 0.6371
        if hp >= 0.6371:
            hp = 0.6371
    elif (Y2[i] - Y9[i]) / h > 1.0:
        hp = (((Y2[i] - Y9[i]) - h) / h) * -0.7068 * 2
        if hp <= -0.7068:
            hp = -0.7068
    HeadPitch.append(hp)

# 3D coordinates
xyz = {'LeftShoulder': np.column_stack([X1, Y1, [0 for z in range(len(X1))]]),
       'RightShoulder': np.column_stack([X2, Y2, [0 for z in range(len(X2))]]),
       'LeftElbow': np.column_stack([X3, Y3, Z3]), 'RightElbow': np.column_stack([X4, Y4, Z4]),
       'LeftWrist': np.column_stack([X5, Y5, Z5]), 'RightWrist': np.column_stack([X6, Y6, Z6])}

# 3D vectors
LS_LE_3D = xyz[landmark3] - xyz[landmark1]
RS_RE_3D = xyz[landmark7] - xyz[landmark5]

LE_LS_3D = xyz[landmark1] - xyz[landmark3]
LW_LE_3D = xyz[landmark3] - xyz[landmark4]

RE_RS_3D = xyz[landmark5] - xyz[landmark7]
RW_RE_3D = xyz[landmark7] - xyz[landmark8]

UpperArmLeft = xyz[landmark3] - xyz[landmark1]
UpperArmRight = xyz[landmark7] - xyz[landmark5]

ZeroXLeft = xyz[landmark3] - xyz[landmark1]
ZeroXRight = xyz[landmark7] - xyz[landmark5]

ZeroXLeft[:, 0] = 0
ZeroXRight[:, 0] = 0

UpperArmLeft[:, 1] = 0
UpperArmRight[:, 1] = 0


# Calculate the left shoulder roll angles
for i in range(LS_LE_3D.shape[0]):
    temp1 = (np.dot(LS_LE_3D[i, :], ZeroXLeft[i, :])) / (np.linalg.norm(LS_LE_3D[i, :]) * np.linalg.norm(ZeroXLeft[i, :]))
    temp = np.arccos(temp1)
    if temp >= 1.56:
        temp = 1.56
    if temp <= np.arccos((np.dot(LS_LE_3D[0, :], ZeroXLeft[0, :])) / (np.linalg.norm(LS_LE_3D[0, :]) * np.linalg.norm(ZeroXLeft[0, :]))):
        temp = 0.0
    LShoulderRoll.append(temp)

# Calculate the right shoulder roll angles
for i in range(RS_RE_3D.shape[0]):
    temp1 = (np.dot(RS_RE_3D[i, :], ZeroXRight[i, :])) / (np.linalg.norm(RS_RE_3D[i, :]) * np.linalg.norm(ZeroXRight[i, :]))
    temp = np.arccos(temp1)
    if temp >= 1.56:
        temp = -1.56
    else:
        temp = temp * (-1)
    if temp > -np.arccos((np.dot(RS_RE_3D[0, :], ZeroXRight[0, :])) / (np.linalg.norm(RS_RE_3D[0, :]) * np.linalg.norm(ZeroXRight[0, :]))):
        temp = 0.0
    RShoulderRoll.append(temp)

# Calculate the left elbow roll angles
for i in range(LE_LS_3D.shape[0]):
    temp1 = (np.dot(LE_LS_3D[i, :], LW_LE_3D[i, :])) / (np.linalg.norm(LE_LS_3D[i, :]) * np.linalg.norm(LW_LE_3D[i, :]))
    temp = np.arccos(temp1)
    if temp >= 1.56:
        temp = -1.56
    else:
        temp = temp * -1
    LElbowRoll.append(temp)

# Calculate the right elbow roll angles
for i in range(RE_RS_3D.shape[0]):
    temp1 = (np.dot(RE_RS_3D[i, :], RW_RE_3D[i, :])) / (np.linalg.norm(RE_RS_3D[i, :]) * np.linalg.norm(RW_RE_3D[i, :]))
    temp = np.arccos(temp1)
    if temp >= 1.56:
        temp = 1.56
    RElbowRoll.append(temp)

# Calculate the left shoulder pitch & left elbow yaw angles
for i in range(LE_LS_3D.shape[0]):
    temp1 = (np.dot(UpperArmLeft[i, :], LS_LE_3D[i, :])) / (np.linalg.norm(UpperArmLeft[i, :]) * np.linalg.norm(LS_LE_3D[i, :]))
    temp = np.arccos(temp1)
    if temp >= np.pi / 2:
        temp = np.pi / 2
    if Y1[i] > Y3[i]:
        temp = temp * -1
    LShoulderPitch.append(temp)

    if LShoulderRoll[i] <= 0.4:
        ley = -np.pi / 2
    elif Y3[i] - Y5[i] > 0.2 * l2_left:
        ley = -np.pi / 2
    elif Y3[i] - Y5[i] < 0 and -(Y3[i] - Y5[i]) > 0.2 * l2_left and LShoulderRoll[i] > 0.7:
        ley = np.pi / 2
    else:
        ley = 0.0
    LElbowYaw.append(ley)

# Calculate the right shoulder pitch & right elbow yaw angles
for i in range(RE_RS_3D.shape[0]):
    temp1 = (np.dot(UpperArmRight[i, :], RS_RE_3D[i, :])) / (np.linalg.norm(UpperArmRight[i, :]) * np.linalg.norm(RS_RE_3D[i, :]))
    temp = np.arccos(temp1)
    if temp >= np.pi / 2:
        temp = np.pi / 2
    if Y2[i] > Y4[i]:
        temp = temp * -1
    RShoulderPitch.append(temp)

    if RShoulderRoll[i] >= -0.4:
        rey = np.pi / 2
    elif Y4[i] - Y6[i] > 0.2 * l2_right:
        rey = np.pi / 2
    elif Y4[i] - Y6[i] < 0 and -(Y4[i] - Y6[i]) > 0.2 * l2_right and RShoulderRoll[i] < -0.7:
        rey = -np.pi / 2
    else:
        rey = 0.0
    RElbowYaw.append(rey)

# Total no. of seconds elapsed in execution
delta = TimeStamp[-1] - TimeStamp[0]
sec = delta.total_seconds()
num = sec / (len(TimeStamp))

timestamps = []

T = []
LSP = []
RSP = []
REY = []
LEY = []
LSR = []
LER = []
RSR = []
RER = []
HY = []
HP = []
HPP = []
HPR = []

# Human joint angles
AngleHuman = [LShoulderPitch, RShoulderPitch, RElbowYaw, LElbowYaw, LShoulderRoll, LElbowRoll,
              RShoulderRoll, RElbowRoll, HeadYaw, HeadPitch, HipRoll, HipPitch]

# Robot joint angles
AngleRobot = [LSP, RSP, REY, LEY, LSR, LER, RSR, RER, HY, HP, HPR, HPP]

# Shorten the timestamps by a scaling factor s
# s is preferable to be between 10 and 15
s = 12
for i in range(len(TimeStamp)):
    s = (num * i) + num
    timestamps.append(s)
n = int((len(timestamps) - (len(timestamps) % s)) / s)
if len(timestamps) % s != 0:
    N = int(n + 1)
else:
    N = n

for j in range(1, N):
    t = timestamps[(j + 1) * s - s]
    T.append(t)
if len(timestamps) % s != 0:
    T.append(T[-1] + T[0])

# Shorten the joint angles
def ShortenData(AngleHuman, AngleRobot):
    for i in range(len(AngleRobot)):
        for j in range(1, N):
            theta = AngleHuman[i][(j + 1) * s - s]
            AngleRobot[i].append(theta)
        if len(timestamps) % s != 0:
            AngleRobot[i].append(AngleHuman[i][-1])

ShortenData(AngleHuman, AngleRobot)

def SafePosition(T, AngleRobot):
    T.append(T[-1] + 2)  # 2 seconds to reach the safe initial position

    for i in range(4, len(AngleRobot)):
        AngleRobot[i].append(0.0)

    for i in range(3):
        AngleRobot[i].append(np.pi / 2)

    AngleRobot[3].append(-np.pi / 2)


SafePosition(T, AngleRobot)

# Export data to input to Pepper
angles = {'TimeStamp': T, 'LShoulderRoll': LSR, 'LElbowRoll': LER,
          'RShoulderRoll': RSR, 'RElbowRoll': RER, 'HeadYaw': HY,
          'HeadPitch': HP, 'LShoulderPitch': LSP, 'RShoulderPitch': RSP,
          'LElbowYaw': LEY, 'RElbowYaw': REY, 'HipRoll': HPR, 'HipPitch': HPP}

ThetaR = pd.DataFrame.from_dict(angles)
ThetaR.to_csv('#file.csv', index=False)

