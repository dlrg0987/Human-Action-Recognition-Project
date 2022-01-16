# importing relevant libraries
import numpy as np
from skimage.transform import resize
from bio_rotate_ja import *

def JEA_create(skel_body, Bio_joint):

    # JEAs feature extraction algorithm (JEA = Joint Euler Angle)

    '''
    # importing relevant libraries
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from skimage.transform import resize
    from PIL import Image
    '''

    dims = skel_body.shape # dimensions of numpy array skel_body

    ### Setting up bio-skeleton body-part vectors
    #bio_body_parts_ja = Bio_body_parts[:,0:3,0:2] 


    bio_joint_constraints = np.array([[[0,0],[0,0],[0,0]], # J0 rotation, body part J0 to J1 #
                                      [[0,0],[0,0],[0,0]], # J0 rotation, body part J0 to J12 (originally was [[0,0],[0,0],[0,-75]]) #
                                      [[0,0],[0,0],[0,0]], # J0 rotation, body part J0 to J16 (originally was [[0,0],[0,0],[0,75]]) #
                                      [[45,-45],[85,-40],[35,-35]], # J1 #
                                      [[0,0],[0,0],[0,0]], # J20 rotation, body part J20 to J2 #
                                      [[45,-45],[35,-35],[80,-80]], # J2 #
                                      [[20,-20],[0,0],[20,-20]], # J20 rotation, body part J20 to J4 #
                                      [[40,-180],[40,-170],[130,-50]], # J4 #
                                      [[0,0],[0,-170],[0,0]], # J5 #
                                      [[20,-20],[0,0],[20,-20]], # J20 rotation, body part J20 to J8 #
                                      [[180,-40],[170,-40],[50,-130]], # J8 #
                                      [[0,0],[0,-170],[0,0]], # J9 #
                                      [[30,-50],[10,-140],[20,-80]], # J12 #
                                      [[0,0],[0,170],[0,0]], # J13 #
                                      [[0,0],[30,-50],[0,0]], # J14 #
                                      [[50,-30],[10,-140],[80,-20]], # J16 #
                                      [[0,0],[0,170],[0,0]], # J17 #
                                      [[0,0],[30,-50],[0,0]]]) # J18 (originally was [[0,0],[30,-50],[0,0]]]) #
                                      
                                      
    x_angle_data = np.zeros((18,dims[0])) # stores data for x axis angles for every joint of every frame
    y_angle_data = np.zeros((18,dims[0])) # stores data for y axis angles for every joint of every frame
    z_angle_data = np.zeros((18,dims[0])) # stores data for z axis angles for every joint of every frame

    ### Run Joint Angle correction algorothm for all frames 'i' in skeleton
    for i in range((dims[0])): # run algorithm for all skeleton frames i

        ### Setting up raw-skeleton body-parts
        """ Raw-skeleton body parts must match body parts of bio-skeleton, in raw skeleton data x coordinate
        is skel_body[i,j,2], y coordinate is skel_body[i,j,0] and z coordinate is skel_body[i,j,1] for frame i and joint j
        """
          
        body_parts_raw = np.array([[[skel_body[i,1,2],skel_body[i,1,0],skel_body[i,1,1]],[skel_body[i,0,2],skel_body[i,0,0],skel_body[i,0,1]]], #
                                    [[skel_body[i,0,2],skel_body[i,0,0],skel_body[i,0,1]],[skel_body[i,16,2],skel_body[i,16,0],skel_body[i,16,1]]], #
                                    [[skel_body[i,0,2],skel_body[i,0,0],skel_body[i,0,1]],[skel_body[i,12,2],skel_body[i,12,0],skel_body[i,12,1]]], #
                                    [[skel_body[i,20,2],skel_body[i,20,0],skel_body[i,20,1]],[skel_body[i,1,2],skel_body[i,1,0],skel_body[i,1,1]]], #
                                    [[skel_body[i,2,2],skel_body[i,2,0],skel_body[i,2,1]],[skel_body[i,20,2],skel_body[i,20,0],skel_body[i,20,1]]], #
                                    [[skel_body[i,3,2],skel_body[i,3,0],skel_body[i,3,1]],[skel_body[i,2,2],skel_body[i,2,0],skel_body[i,2,1]]], #
                                    [[skel_body[i,20,2],skel_body[i,20,0],skel_body[i,20,1]],[skel_body[i,8,2],skel_body[i,8,0],skel_body[i,8,1]]], #
                                    [[skel_body[i,8,2],skel_body[i,8,0],skel_body[i,8,1]],[skel_body[i,9,2],skel_body[i,9,0],skel_body[i,9,1]]], #
                                    [[skel_body[i,9,2],skel_body[i,9,0],skel_body[i,9,1]],[skel_body[i,10,2],skel_body[i,10,0],skel_body[i,10,1]]], #
                                    [[skel_body[i,20,2],skel_body[i,20,0],skel_body[i,20,1]],[skel_body[i,4,2],skel_body[i,4,0],skel_body[i,4,1]]], #
                                    [[skel_body[i,4,2],skel_body[i,4,0],skel_body[i,4,1]],[skel_body[i,5,2],skel_body[i,5,0],skel_body[i,5,1]]], #
                                    [[skel_body[i,5,2],skel_body[i,5,0],skel_body[i,5,1]],[skel_body[i,6,2],skel_body[i,6,0],skel_body[i,6,1]]], #
                                    [[skel_body[i,16,2],skel_body[i,16,0],skel_body[i,16,1]],[skel_body[i,17,2],skel_body[i,17,0],skel_body[i,17,1]]], #
                                    [[skel_body[i,17,2],skel_body[i,17,0],skel_body[i,17,1]],[skel_body[i,18,2],skel_body[i,18,0],skel_body[i,18,1]]], #
                                    [[skel_body[i,18,2],skel_body[i,18,0],skel_body[i,18,1]],[skel_body[i,19,2],skel_body[i,19,0],skel_body[i,19,1]]], #
                                    [[skel_body[i,12,2],skel_body[i,12,0],skel_body[i,12,1]],[skel_body[i,13,2],skel_body[i,13,0],skel_body[i,13,1]]], #
                                    [[skel_body[i,13,2],skel_body[i,13,0],skel_body[i,13,1]],[skel_body[i,14,2],skel_body[i,14,0],skel_body[i,14,1]]], #
                                    [[skel_body[i,14,2],skel_body[i,14,0],skel_body[i,14,1]],[skel_body[i,15,2],skel_body[i,15,0],skel_body[i,15,1]]]]) #

        ### Rotating Bio_skeleton to face direction of raw_skeleton frame 'i'. This ensures that vectors obtained are pointing in correct direction for Bio_skeleton
          
        Bio_body_parts_ja = bio_rotate_ja(skel_body, Bio_joint, i) # function that rotates bio-skeleton in z axis to match direction of ith frame in raw-skeleton
         
        ### Obtaining raw-skeleton body-part vectors and bio-skeleton body part verctors for frame 'i'
          
        body_part_vectors_bio = np.array([np.subtract(Bio_body_parts_ja[0,:,0],Bio_body_parts_ja[0,:,1]),
                                          np.subtract(Bio_body_parts_ja[1,:,1],Bio_body_parts_ja[1,:,0]),
                                          np.subtract(Bio_body_parts_ja[2,:,1],Bio_body_parts_ja[2,:,0]),
                                          np.subtract(Bio_body_parts_ja[3,:,0],Bio_body_parts_ja[3,:,1]),
                                          np.subtract(Bio_body_parts_ja[4,:,0],Bio_body_parts_ja[4,:,1]),
                                          np.subtract(Bio_body_parts_ja[5,:,0],Bio_body_parts_ja[5,:,1]),
                                          np.subtract(Bio_body_parts_ja[6,:,1],Bio_body_parts_ja[6,:,0]),
                                          np.subtract(Bio_body_parts_ja[7,:,1],Bio_body_parts_ja[7,:,0]),
                                          np.subtract(Bio_body_parts_ja[8,:,1],Bio_body_parts_ja[8,:,0]),
                                          np.subtract(Bio_body_parts_ja[9,:,1],Bio_body_parts_ja[9,:,0]),
                                          np.subtract(Bio_body_parts_ja[10,:,1],Bio_body_parts_ja[10,:,0]),
                                          np.subtract(Bio_body_parts_ja[11,:,1],Bio_body_parts_ja[11,:,0]),
                                          np.subtract(Bio_body_parts_ja[12,:,1],Bio_body_parts_ja[12,:,0]),
                                          np.subtract(Bio_body_parts_ja[13,:,1],Bio_body_parts_ja[13,:,0]),
                                          np.subtract(Bio_body_parts_ja[14,:,1],Bio_body_parts_ja[14,:,0]),
                                          np.subtract(Bio_body_parts_ja[15,:,1],Bio_body_parts_ja[15,:,0]),
                                          np.subtract(Bio_body_parts_ja[16,:,1],Bio_body_parts_ja[16,:,0]),
                                          np.subtract(Bio_body_parts_ja[17,:,1],Bio_body_parts_ja[17,:,0])])
          
          
        for j in range((dims[1] - 7)): # Loop for all body-parts vectors j in frame i
            body_part_vectors_raw = np.array([np.subtract(body_parts_raw[0,0,:],body_parts_raw[0,1,:]),
                                              np.subtract(body_parts_raw[1,1,:],body_parts_raw[1,0,:]),
                                              np.subtract(body_parts_raw[2,1,:],body_parts_raw[2,0,:]),
                                              np.subtract(body_parts_raw[3,0,:],body_parts_raw[3,1,:]),
                                              np.subtract(body_parts_raw[4,0,:],body_parts_raw[4,1,:]),
                                              np.subtract(body_parts_raw[5,0,:],body_parts_raw[5,1,:]),
                                              np.subtract(body_parts_raw[6,1,:],body_parts_raw[6,0,:]),
                                              np.subtract(body_parts_raw[7,1,:],body_parts_raw[7,0,:]),
                                              np.subtract(body_parts_raw[8,1,:],body_parts_raw[8,0,:]),
                                              np.subtract(body_parts_raw[9,1,:],body_parts_raw[9,0,:]),
                                              np.subtract(body_parts_raw[10,1,:],body_parts_raw[10,0,:]),
                                              np.subtract(body_parts_raw[11,1,:],body_parts_raw[11,0,:]),
                                              np.subtract(body_parts_raw[12,1,:],body_parts_raw[12,0,:]),
                                              np.subtract(body_parts_raw[13,1,:],body_parts_raw[13,0,:]),
                                              np.subtract(body_parts_raw[14,1,:],body_parts_raw[14,0,:]),
                                              np.subtract(body_parts_raw[15,1,:],body_parts_raw[15,0,:]),
                                              np.subtract(body_parts_raw[16,1,:],body_parts_raw[16,0,:]),
                                              np.subtract(body_parts_raw[17,1,:],body_parts_raw[17,0,:])])
            

            bp_j_vec_raw = np.transpose(body_part_vectors_raw[j,:].reshape(1,3))
            bp_j_vec_bio = np.transpose(body_part_vectors_bio[j,:].reshape(1,3)) 

            # normalising bp_j_vec_bio to match length of bp_j_vec_raw
            ####bp_j_vec_raw = bp_j_vec_raw /np.linalg.norm(bp_j_vec_raw)
            bp_j_vec_bio = (bp_j_vec_bio /np.linalg.norm(bp_j_vec_bio)) * (np.linalg.norm(bp_j_vec_raw))

            # solving for error numpy.linalg.LinAlgError: SVD did not converge caused by inf and nan values in matrix within np.linalg.pinv in line below
            bp_mat_bio = np.matmul(bp_j_vec_bio, np.transpose(bp_j_vec_bio))
            bp_mat_raw = np.matmul(bp_j_vec_raw, np.transpose(bp_j_vec_bio))
            
            #bp_mat_bio = bp_mat_bio.replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna()
            #bp_mat_raw = bp_mat_raw.replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna()

            bp_mat_bio = np.nan_to_num(bp_mat_bio, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            bp_mat_raw = np.nan_to_num(bp_mat_raw, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Obtain rotation matrix rotation bp_j_vec_bio to bp_j_vec_raw
            A = np.dot(bp_mat_raw,np.linalg.pinv(bp_mat_bio))
            
            # α = arctan(-a23/a33) = rotation angle about x axis for bio to raw skeleton 
            alpha_JA = np.arctan2((-1*A[1,2]),A[2,2])  
            # β = arctan(-a13/sqrt(a11^2 + a12^2)) = rotation angle about y axis for bio to raw skeleton
            beta_JA = np.arctan2((A[0,2]),(np.sqrt(A[0,0]**2 + A[0,1]**2)))
            # γ = arctan(-a12/a11) = rotation angle about z axis for bio-bp to raw-bp
            gamma_JA = np.arctan2((-1*A[0,1]),A[0,0])

            # Convert angles to degrees from radians 
            gamma_JA = (gamma_JA/(2*np.pi))*360
            beta_JA  = (beta_JA /(2*np.pi))*360
            alpha_JA = (alpha_JA/(2*np.pi))*360
            
            # storing angle value for all joints and frames 
            x_angle_data[j,i] = alpha_JA
            y_angle_data[j,i] = beta_JA
            z_angle_data[j,i] = gamma_JA


    JEA_im_array = np.zeros((224,224,3))
    #print(x_angle_data.shape)
    #print(y_angle_data.shape)
    #print(z_angle_data.shape)
    

    '''
    # converting top Uint8 for all n_angle_data's
    x_angle_data = 255*x_angle_data
    y_angle_data = 255*y_angle_data
    z_angle_data = 255*z_angle_data
    '''

    # Resize all angle data to 224 by 224
    x_angle_data = resize(x_angle_data, (224, 224))
    y_angle_data = resize(y_angle_data, (224, 224))
    z_angle_data = resize(z_angle_data, (224, 224))

    # Taking absolute value of all n_angle_data
    x_angle_data = np.absolute(x_angle_data)
    y_angle_data = np.absolute(y_angle_data)
    z_angle_data = np.absolute(z_angle_data)
    
    # Normalising n_angle_data's so that all values are between 0 - 1
    lam_x = np.amax(x_angle_data)
    x_angle_data = np.dot((1/lam_x),x_angle_data)

    lam_y = np.amax(y_angle_data)
    y_angle_data = np.dot((1/lam_y),y_angle_data)

    lam_z = np.amax(z_angle_data)
    z_angle_data = np.dot((1/lam_z),z_angle_data)


    JEA_im_array[:,:,0] = x_angle_data
    JEA_im_array[:,:,1] = y_angle_data
    JEA_im_array[:,:,2] = z_angle_data

    #JEA_im_array = np.absolute(JEA_im_array)
    #print(np.amax(JEA_im_array))

    return JEA_im_array
