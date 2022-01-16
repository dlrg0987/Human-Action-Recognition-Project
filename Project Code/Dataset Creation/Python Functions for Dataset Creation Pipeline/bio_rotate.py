### General Bio-skeleton rotation function (rotates bio-skeleton about z axis to match direction of frame 'i' of raw skeleton)
# Importing Libraries
import numpy as np
  
def bio_rotate(skel_body, Bio_joint, frame):
  '''
  # Importing Libraries
  import numpy as np
  '''

  dims = skel_body.shape
  
  ### normalizing joints of raw skeleton:
  for i in range((dims[0])):
    x_mean = np.mean(skel_body[i,:,0])
    y_mean = np.mean(skel_body[i,:,1])
    z_mean = np.mean(skel_body[i,:,2])

    skel_body[i,:,0] = np.divide(skel_body[i,:,0],x_mean)
    skel_body[i,:,1] = np.divide(skel_body[i,:,1],y_mean)
    skel_body[i,:,2] = np.divide(skel_body[i,:,2],z_mean)

  raw_skel_dim = skel_body.shape # dimensions of numpy array skel_body
  #print(raw_skel_dim)

  bio_skel_dim = Bio_joint.shape
  #print(bio_skel_dim)

  raw_skel_joint = np.array([[skel_body[frame,0,2], skel_body[frame,0,0], skel_body[frame,0,1]]
                            ,[skel_body[frame,1,2], skel_body[frame,1,0], skel_body[frame,1,1]]
                            ,[skel_body[frame,2,2], skel_body[frame,2,0], skel_body[frame,2,1]]
                            ,[skel_body[frame,3,2], skel_body[frame,3,0], skel_body[frame,3,1]]
                            ,[skel_body[frame,8,2], skel_body[frame,8,0], skel_body[frame,8,1]]
                            ,[skel_body[frame,9,2], skel_body[frame,9,0], skel_body[frame,9,1]]
                            ,[skel_body[frame,10,2], skel_body[frame,10,0], skel_body[frame,10,1]]
                            ,[skel_body[frame,4,2], skel_body[frame,4,0], skel_body[frame,4,1]]
                            ,[skel_body[frame,5,2], skel_body[frame,5,0], skel_body[frame,5,1]]
                            ,[skel_body[frame,6,2], skel_body[frame,6,0], skel_body[frame,6,1]]
                            ,[skel_body[frame,16,2], skel_body[frame,16,0], skel_body[frame,16,1]]
                            ,[skel_body[frame,17,2], skel_body[frame,17,0], skel_body[frame,17,1]]
                            ,[skel_body[frame,18,2], skel_body[frame,18,0], skel_body[frame,18,1]]
                            ,[skel_body[frame,19,2], skel_body[frame,19,0], skel_body[frame,19,1]]
                            ,[skel_body[frame,12,2], skel_body[frame,12,0], skel_body[frame,12,1]]
                            ,[skel_body[frame,13,2], skel_body[frame,13,0], skel_body[frame,13,1]]
                            ,[skel_body[frame,14,2], skel_body[frame,14,0], skel_body[frame,14,1]]
                            ,[skel_body[frame,15,2], skel_body[frame,15,0], skel_body[frame,15,1]]
                            ,[skel_body[frame,20,2], skel_body[frame,20,0], skel_body[frame,20,1]]])

  #print(raw_skel_joint.shape)

  Bio_joint = np.transpose(Bio_joint) # joint coords of bio skeleton (make size = 3 x 19)
  raw_skel_joint = np.transpose(raw_skel_joint) # joint coords of raw skeleton (make size = 3 x 19)
  #print(raw_skel_joint.shape)
  #print(Bio_joint.shape)

  """A = (body_part_vector_t_m * TRANSPOSE(body_part_vector_bio_m)) * INVERSE(body_part_vector_bio_m * TRANSPOSE(body_part_vector_bio_m)) """
  # solving for error numpy.linalg.LinAlgError: SVD did not converge caused by inf and nan values in matrix within np.linalg.pinv in line below
  j_mat_bio = np.matmul(Bio_joint,np.transpose(Bio_joint))
  j_mat_raw = np.matmul(raw_skel_joint,np.transpose(Bio_joint))

  j_mat_bio = np.nan_to_num(j_mat_bio, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
  j_mat_raw = np.nan_to_num(j_mat_raw, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            

  # Rotation matrix that transforms bio-skeleton to raw-skeleton (1st frame)
  rot_mat = np.dot((np.linalg.pinv(j_mat_bio)),(j_mat_raw))
  #print(rot_mat.shape)

  # See following link to rotation matrices notation + derivation: http://planning.cs.uiuc.edu/node103.html and http://planning.cs.uiuc.edu/node102.html
  gamma = np.arctan2((-1*rot_mat[0,1]),rot_mat[0,0]) # γ = arctan(-a12/a11) = rotation angle about z axis for bio to raw skeleton

  z_rot_mat = np.array([[np.cos(gamma), -1*np.sin(gamma), 0], [np.sin(gamma), np.cos(gamma), 0], [0, 0, 1]]) # Z axis rotation matrix for bio skeleton

  # Determining x and y rotation angles
  beta = np.arctan2((rot_mat[0,2]),(np.sqrt(rot_mat[0,0]**2 + rot_mat[0,1]**2))) # β = arctan(-a13/sqrt(a11^2 + a12^2)) = rotation angle about y axis for bio to raw skeleton
  alpha = np.arctan2((-1*rot_mat[1,2]),rot_mat[2,2]) # α = arctan(-a23/a33) = rotation angle about x axis for bio to raw skeleton

  y_rot_mat = np.array([[np.cos(beta), 0, np.sin(beta)], [0, 1, 0], [-1*np.sin(beta), 0, np.cos(beta)]]) # y axis rotation matrix for bio skeleton
  x_rot_mat = np.array([[1, 0, 0], [0, np.cos(alpha), -1*np.sin(alpha)], [0, np.sin(alpha), np.cos(alpha)]]) # x axis rotation matrix for bio skeleton

  # Rotating bio-skeleton about z-axis by angle gamma
  Bio_joint = np.transpose(np.matmul(z_rot_mat, Bio_joint)) 

  # Rotating bio-skeleton about y-axis by angle beta
  #Bio_joint = np.matmul(y_rot_mat, Bio_joint)

  # Rotating bio-skeleton about x-axis by angle alpha
  #Bio_joint = np.transpose(np.matmul(x_rot_mat, Bio_joint))
  #print(Bio_joint)
  #print(Bio_joint.shape)


  ### Re-defining Bio-Skeleton to orient it with first frame of raw skeleton
  # Bio-Skeleton body_part_1 (Bio_joints 3 and 2)
  x_b1 = np.array([Bio_joint[3,0], Bio_joint[2,0]])
  y_b1 = np.array([Bio_joint[3,1], Bio_joint[2,1]])
  z_b1 = np.array([Bio_joint[3,2], Bio_joint[2,2]])
  joint_ID_b1 = np.array([3,2])

  # Bio-Skeleton body_part_2 (Bio_joints 20 and 2)
  x_b2 = np.array([Bio_joint[2,0], Bio_joint[18,0]])
  y_b2 = np.array([Bio_joint[2,1], Bio_joint[18,1]])
  z_b2 = np.array([Bio_joint[2,2], Bio_joint[18,2]])
  joint_ID_b2 = np.array([2,18])

  # Bio-Skeleton body_part_3 (Bio_joints 20 and 4)
  x_b3 = np.array([Bio_joint[18,0], Bio_joint[4,0]])
  y_b3 = np.array([Bio_joint[18,1], Bio_joint[4,1]])
  z_b3 = np.array([Bio_joint[18,2], Bio_joint[4,2]])
  joint_ID_b3 = np.array([18,4])

  # Bio-Skeleton body_part_4 (Bio_joints 4 and 5)
  x_b4 = np.array([Bio_joint[4,0], Bio_joint[5,0]])
  y_b4 = np.array([Bio_joint[4,1], Bio_joint[5,1]])
  z_b4 = np.array([Bio_joint[4,2], Bio_joint[5,2]])
  joint_ID_b4 = np.array([4,5])

  # Bio-Skeleton body_part_5 (Bio_joints 5 and 6)
  x_b5 = np.array([Bio_joint[5,0], Bio_joint[6,0]])
  y_b5 = np.array([Bio_joint[5,1], Bio_joint[6,1]])
  z_b5 = np.array([Bio_joint[5,2], Bio_joint[6,2]])
  joint_ID_b5 = np.array([5,6])

  # Bio-Skeleton body_part_6 (Bio_joints 20 and 8)
  x_b6 = np.array([Bio_joint[18,0], Bio_joint[7,0]])
  y_b6 = np.array([Bio_joint[18,1], Bio_joint[7,1]])
  z_b6 = np.array([Bio_joint[18,2], Bio_joint[7,2]])
  joint_ID_b6 = np.array([18,7])

  # Bio-Skeleton body_part_7 (Bio_joints 8 and 9)
  x_b7 = np.array([Bio_joint[7,0], Bio_joint[8,0]])
  y_b7 = np.array([Bio_joint[7,1], Bio_joint[8,1]])
  z_b7 = np.array([Bio_joint[7,2], Bio_joint[8,2]])
  joint_ID_b7 = np.array([7,8])

  # Bio-Skeleton body_part_8 (Bio_joints 9 and 10)
  x_b8 = np.array([Bio_joint[8,0], Bio_joint[9,0]])
  y_b8 = np.array([Bio_joint[8,1], Bio_joint[9,1]])
  z_b8 = np.array([Bio_joint[8,2], Bio_joint[9,2]])
  joint_ID_b8 = np.array([8,9])

  # Bio-Skeleton body_part_9 (Bio_joints 20 and 1)
  x_b9 = np.array([Bio_joint[18,0], Bio_joint[1,0]])
  y_b9 = np.array([Bio_joint[18,1], Bio_joint[1,1]])
  z_b9 = np.array([Bio_joint[18,2], Bio_joint[1,2]])
  joint_ID_b9 = np.array([18,1])

  # Bio-Skeleton body_part_10 (Bio_joints 1 and 0)
  x_b10 = np.array([Bio_joint[1,0], Bio_joint[0,0]])
  y_b10 = np.array([Bio_joint[1,1], Bio_joint[0,1]])
  z_b10 = np.array([Bio_joint[1,2], Bio_joint[0,2]])
  joint_ID_b10 = np.array([1,0])

  # Bio-Skeleton body_part_11 (Bio_joints 0 and 12)
  x_b11 = np.array([Bio_joint[0,0], Bio_joint[10,0]])
  y_b11 = np.array([Bio_joint[0,1], Bio_joint[10,1]])
  z_b11 = np.array([Bio_joint[0,2], Bio_joint[10,2]])
  joint_ID_b11 = np.array([0,10])

  # Bio-Skeleton body_part_12 (Bio_joints 0 and 16)
  x_b12 = np.array([Bio_joint[0,0], Bio_joint[14,0]])
  y_b12 = np.array([Bio_joint[0,1], Bio_joint[14,1]])
  z_b12 = np.array([Bio_joint[0,2], Bio_joint[14,2]])
  joint_ID_b12 = np.array([0,14])

  # Bio-Skeleton body_part_13 (Bio_joints 12 and 13)
  x_b13 = np.array([Bio_joint[10,0], Bio_joint[11,0]])
  y_b13 = np.array([Bio_joint[10,1], Bio_joint[11,1]])
  z_b13 = np.array([Bio_joint[10,2], Bio_joint[11,2]])
  joint_ID_b13 = np.array([10,11])

  # Bio-Skeleton body_part_14 (Bio_joints 13 and 14)
  x_b14 = np.array([Bio_joint[11,0], Bio_joint[12,0]])
  y_b14 = np.array([Bio_joint[11,1], Bio_joint[12,1]])
  z_b14 = np.array([Bio_joint[11,2], Bio_joint[12,2]])
  joint_ID_b14 = np.array([11,12])

  # Bio-Skeleton body_part_15 (Bio_joints 14 and 15)
  x_b15 = np.array([Bio_joint[12,0], Bio_joint[13,0]])
  y_b15 = np.array([Bio_joint[12,1], Bio_joint[13,1]])
  z_b15 = np.array([Bio_joint[12,2], Bio_joint[13,2]])
  joint_ID_b15 = np.array([12,13])

  # Bio-Skeleton body_part_16 (Bio_joints 16 and 17)
  x_b16 = np.array([Bio_joint[14,0], Bio_joint[15,0]])
  y_b16 = np.array([Bio_joint[14,1], Bio_joint[15,1]])
  z_b16 = np.array([Bio_joint[14,2], Bio_joint[15,2]])
  joint_ID_b16 = np.array([14,15])

  # Bio-Skeleton body_part_17 (Bio_joints 17 and 18)
  x_b17 = np.array([Bio_joint[15,0], Bio_joint[16,0]])
  y_b17 = np.array([Bio_joint[15,1], Bio_joint[16,1]])
  z_b17 = np.array([Bio_joint[15,2], Bio_joint[16,2]])
  joint_ID_b17 = np.array([15,16])

  # Bio-Skeleton body_part_18 (Bio_joints 18 and 19)
  x_b18 = np.array([Bio_joint[16,0], Bio_joint[17,0]])
  y_b18 = np.array([Bio_joint[16,1], Bio_joint[17,1]])
  z_b18 = np.array([Bio_joint[16,2], Bio_joint[17,2]])
  joint_ID_b18 = np.array([16,17])

  
  Bio_body_parts = np.array([[x_b1,y_b1,z_b1,joint_ID_b1],[x_b2,y_b2,z_b2,joint_ID_b2],
                            [x_b3,y_b3,z_b3,joint_ID_b3],[x_b4,y_b4,z_b4,joint_ID_b4],
                            [x_b5,y_b5,z_b5,joint_ID_b5],[x_b6,y_b6,z_b6,joint_ID_b6],
                            [x_b7,y_b7,z_b7,joint_ID_b7],[x_b8,y_b8,z_b8,joint_ID_b8],
                            [x_b9,y_b9,z_b9,joint_ID_b9],[x_b10,y_b10,z_b10,joint_ID_b10],
                            [x_b11,y_b11,z_b11,joint_ID_b11],[x_b12,y_b12,z_b12,joint_ID_b12],
                            [x_b13,y_b13,z_b13,joint_ID_b13],[x_b14,y_b14,z_b14,joint_ID_b14],
                            [x_b15,y_b15,z_b15,joint_ID_b15],[x_b16,y_b16,z_b16,joint_ID_b16],
                            [x_b17,y_b17,z_b17,joint_ID_b17],[x_b18,y_b18,z_b18,joint_ID_b18]])
  
  """
  Bio_body_parts_ja = np.array([[x_b10,y_b10,z_b10],[x_b11,y_b11,z_b11], # rotates about joint 1, joint 0
                                [x_b12,y_b12,z_b12],[x_b9,y_b9,z_b9], # rotates about joint 0, joint 1
                                [x_b2,y_b2,z_b2],[x_b1,y_b1,z_b1], # rotates about joint 1 , joint 1
                                [x_b3,y_b3,z_b3],[x_b4,y_b4,z_b4], # rotates about joint 0, joint 0
                                [x_b5,y_b5,z_b5],[x_b6,y_b6,z_b6], # rotates about joint 0 , joint 0
                                [x_b7,y_b7,z_b7],[x_b8,y_b8,z_b8], # rotates about joint 0, joint 0
                                [x_b13,y_b13,z_b13],[x_b14,y_b14,z_b14], # rotates about joint 0 , joint 0
                                [x_b15,y_b15,z_b15],[x_b16,y_b16,z_b16], # rotates about joint 0 , joint 0
                                [x_b17,y_b17,z_b17],[x_b18,y_b18,z_b18]]) # rotates about joint 0 , joint 0
  """
  
  return Bio_body_parts
