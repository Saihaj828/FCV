when the bot stops in patient zone:

`
ros2 topic pub /ServerCommunication synapse_msgs/msg/ServerCommunication "{src: 2, dest: 1, uid: 101, ack: 0, msg: '{LOC: HOSPITAL_1}'}" -1
`

when the bot stops in hospital zone:

`
ros2 topic pub /ServerCommunication synapse_msgs/msg/ServerCommunication "{src: 2, dest: 1, uid: 101, ack: 0, msg: '{LOC: PATIENT_2}'}" -1
`
