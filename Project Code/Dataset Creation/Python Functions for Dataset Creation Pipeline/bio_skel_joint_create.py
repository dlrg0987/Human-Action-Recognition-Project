#### THIS FUNCTION CREATES THE JOINTS OF THE BIO-CONSTRAINED SKELETON
# Importing Libraries
import numpy as np

def bio_skel_joint_create(Bio_lengths):
### Set Up of Bio-Constrained Skeleton Frame:
    '''
    # Importing Libraries
    import numpy as np
    '''
    # Setting up Bio-Skeleton Joint positions
    Bio_joint_0 = np.array([0,0,0]) # Bio_joint_0 located at origin
    Bio_joint_1 = Bio_joint_0 + np.array([0,0,Bio_lengths[9]]) # Bio_joint_1 Position in reference to Global Coordinate Frame
    Bio_joint_20 = Bio_joint_1 + np.array([0,0,Bio_lengths[8]]) # Bio_joint_20 Position in reference to Global Coordinate Frame
    Bio_joint_2 = Bio_joint_20 + np.array([0,0,Bio_lengths[1]]) # Bio_joint_2 Position in reference to Global Coordinate Frame
    Bio_joint_3 = Bio_joint_2 + np.array([0,0,Bio_lengths[0]]) # Bio_joint_3 Position in reference to Global Coordinate Frame
    Bio_joint_4 = Bio_joint_20 + np.array([0,-1*Bio_lengths[2],0]) # Bio_joint_4 Position in reference to Global Coordinate Frame
    Bio_joint_5 = Bio_joint_4 + np.array([0,0,-1*Bio_lengths[3]]) # Bio_joint_5 Position in reference to Global Coordinate Frame
    Bio_joint_6 = Bio_joint_5 + np.array([0,0,-1*Bio_lengths[4]]) # Bio_joint_6 Position in reference to Global Coordinate Frame
    Bio_joint_8 = Bio_joint_20 + np.array([0,Bio_lengths[5],0]) # Bio_joint_8 Position in reference to Global Coordinate Frame
    Bio_joint_9 = Bio_joint_8 + np.array([0,0,-1*Bio_lengths[6]]) # Bio_joint_9 Position in reference to Global Coordinate Frame
    Bio_joint_10 = Bio_joint_9 + np.array([0,0,-1*Bio_lengths[7]]) # Bio_joint_10 Position in reference to Global Coordinate Frame
    angle_1 = (-75/360)*2*np.pi # angle by which Bio_joint_12 will be positioned
    angle_2 = (75/360)*2*np.pi # angle by which Bio_joint_12 will be positioned
    Bio_joint_12 = np.dot(np.array([[1,0,0],[0,np.cos(angle_1),-1*np.sin(angle_1)],[0,np.sin(angle_1),np.cos(angle_1)]]), (Bio_joint_0 + np.array([0,0,-1*Bio_lengths[10]]))) # Bio_joint_12 Position in reference to Global Coordinate Frame
    Bio_joint_16 = np.dot(np.array([[1,0,0],[0,np.cos(angle_2),-1*np.sin(angle_2)],[0,np.sin(angle_2),np.cos(angle_2)]]), (Bio_joint_0 + np.array([0,0,-1*Bio_lengths[11]]))) # Bio_joint_16 Position in reference to Global Coordinate Frame
    Bio_joint_13 = Bio_joint_12 + np.array([0,0,-1*Bio_lengths[12]]) # Bio_joint_13 Position in reference to Global Coordinate Frame
    Bio_joint_14 = Bio_joint_13 + np.array([0,0,-1*Bio_lengths[13]]) # Bio_joint_14 Position in reference to Global Coordinate Frame
    Bio_joint_15 = Bio_joint_14 + np.array([1*Bio_lengths[14],0,0]) # Bio_joint_15 Position in reference to Global Coordinate Frame
    Bio_joint_17 = Bio_joint_16 + np.array([0,0,-1*Bio_lengths[15]]) # Bio_joint_17 Position in reference to Global Coordinate Frame
    Bio_joint_18 = Bio_joint_17 + np.array([0,0,-1*Bio_lengths[16]]) # Bio_joint_18 Position in reference to Global Coordinate Frame
    Bio_joint_19 = Bio_joint_18 + np.array([1*Bio_lengths[17],0,0]) # Bio_joint_19 Position in reference to Global Coordinate Frame

    # All Joints in Bio-Skeleton (in single numpy array of dimensions (19,3), where there are 19 joints and each joint as 3 coords)
    Bio_joint = np.array([Bio_joint_0, Bio_joint_1, Bio_joint_2, Bio_joint_3, Bio_joint_4, Bio_joint_5, Bio_joint_6, Bio_joint_8, Bio_joint_9, Bio_joint_10, Bio_joint_12, Bio_joint_13, Bio_joint_14, Bio_joint_15, Bio_joint_16, Bio_joint_17, Bio_joint_18, Bio_joint_19, Bio_joint_20])


    return Bio_joint
