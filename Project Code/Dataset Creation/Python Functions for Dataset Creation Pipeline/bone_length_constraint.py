### importing relevant libraries
import numpy as np


def bone_length_constraint(skel_body, Bio_body_parts, Bio_lengths):

    ### Bone Length Constraint Algorithm 

    '''
    ### importing relevant libraries
    import numpy as np
    '''

    dims = skel_body.shape # dimensions of numpy array skel_body



    ### normalizing joints of raw skeleton:
    for i in range((dims[0])):
        x_mean = np.mean(skel_body[i,:,0])
        y_mean = np.mean(skel_body[i,:,1])
        z_mean = np.mean(skel_body[i,:,2])

        skel_body[i,:,0] = np.divide(skel_body[i,:,0],x_mean)
        skel_body[i,:,1] = np.divide(skel_body[i,:,1],y_mean)
        skel_body[i,:,2] = np.divide(skel_body[i,:,2],z_mean)


    ### Checking body-part lengths for every raw-skeleton frame
    for i in range((dims[0])): # run algorithm for all skeleton frames i
        # creating joint_adjust_ID variable to store info on whether or not a particular joint has been adjusted
        joint_adjust_ID = np.zeros(19)

        ### Setting up raw-skeleton body-parts
        """ Raw-skeleton body parts must match body parts of bio-skeleton, in raw skeleton data x coordinate
        is skel_body[i,j,2], y coordinate is skel_body[i,j,0] and z coordinate is skel_body[i,j,1] for frame i and joint j"""
        body_parts_raw_i = np.array([[[skel_body[i,3,2],skel_body[i,3,0],skel_body[i,3,1],3],[skel_body[i,2,2],skel_body[i,2,0],skel_body[i,2,1],2]],
                                   [[skel_body[i,2,2],skel_body[i,2,0],skel_body[i,2,1],2],[skel_body[i,20,2],skel_body[i,20,0],skel_body[i,20,1],18]],
                                   [[skel_body[i,20,2],skel_body[i,20,0],skel_body[i,20,1],18],[skel_body[i,8,2],skel_body[i,8,0],skel_body[i,8,1],4]],
                                   [[skel_body[i,8,2],skel_body[i,8,0],skel_body[i,8,1],4],[skel_body[i,9,2],skel_body[i,9,0],skel_body[i,9,1],5]],
                                   [[skel_body[i,9,2],skel_body[i,9,0],skel_body[i,9,1],5],[skel_body[i,10,2],skel_body[i,10,0],skel_body[i,10,1],6]],
                                   [[skel_body[i,20,2],skel_body[i,20,0],skel_body[i,20,1],18],[skel_body[i,4,2],skel_body[i,4,0],skel_body[i,4,1],7]],
                                   [[skel_body[i,4,2],skel_body[i,4,0],skel_body[i,4,1],7],[skel_body[i,5,2],skel_body[i,5,0],skel_body[i,5,1],8]],
                                   [[skel_body[i,5,2],skel_body[i,5,0],skel_body[i,5,1],8],[skel_body[i,6,2],skel_body[i,6,0],skel_body[i,6,1],9]],
                                   [[skel_body[i,20,2],skel_body[i,20,0],skel_body[i,20,1],18],[skel_body[i,1,2],skel_body[i,1,0],skel_body[i,1,1],1]],
                                   [[skel_body[i,1,2],skel_body[i,1,0],skel_body[i,1,1],1],[skel_body[i,0,2],skel_body[i,0,0],skel_body[i,0,1],0]],
                                   [[skel_body[i,0,2],skel_body[i,0,0],skel_body[i,0,1],0],[skel_body[i,16,2],skel_body[i,16,0],skel_body[i,16,1],10]],
                                   [[skel_body[i,0,2],skel_body[i,0,0],skel_body[i,0,1],0],[skel_body[i,12,2],skel_body[i,12,0],skel_body[i,12,1],14]],
                                   [[skel_body[i,16,2],skel_body[i,16,0],skel_body[i,16,1],10],[skel_body[i,17,2],skel_body[i,17,0],skel_body[i,17,1],11]],
                                   [[skel_body[i,17,2],skel_body[i,17,0],skel_body[i,17,1],11],[skel_body[i,18,2],skel_body[i,18,0],skel_body[i,18,1],12]],
                                   [[skel_body[i,18,2],skel_body[i,18,0],skel_body[i,18,1],12],[skel_body[i,19,2],skel_body[i,19,0],skel_body[i,19,1],13]],
                                   [[skel_body[i,12,2],skel_body[i,12,0],skel_body[i,12,1],14],[skel_body[i,13,2],skel_body[i,13,0],skel_body[i,13,1],15]],
                                   [[skel_body[i,13,2],skel_body[i,13,0],skel_body[i,13,1],15],[skel_body[i,14,2],skel_body[i,14,0],skel_body[i,14,1],16]],
                                   [[skel_body[i,14,2],skel_body[i,14,0],skel_body[i,14,1],16],[skel_body[i,15,2],skel_body[i,15,0],skel_body[i,15,1],17]]])
      

        for j in range((dims[1] - 7)): # Loop for body parts j in frame i

            if i == 0: # Check if the raw-skeleton frame being evaluated is the first frame
                body_part_j_raw = body_parts_raw_i[j,:,:] # Obtain body-part j from list of raw-skeleton body parts in array 'body_parts_raw'

                ### Implementing body-part length constraint algorithm:
                # Check length of body part j
                raw_length_j = np.sqrt((body_part_j_raw[0,0] - body_part_j_raw[1,0])**2 +
                                       (body_part_j_raw[0,1] - body_part_j_raw[1,1])**2 + 
                                       (body_part_j_raw[0,2] - body_part_j_raw[1,2])**2)
                if raw_length_j == Bio_lengths[j]:
                    continue
                elif raw_length_j != Bio_lengths[j]:
                    # Euclidean distance between raw_body_part_j_joint_1 and Bio_body_part_j_joint_1
                    L1 = np.sqrt((body_part_j_raw[0,0] - Bio_body_parts[j,0,0])**2 +
                                 (body_part_j_raw[0,1] - Bio_body_parts[j,1,0])**2 + 
                                 (body_part_j_raw[0,2] - Bio_body_parts[j,2,0])**2)
                    # Euclidean distance between raw_body_part_j_joint_2 and Bio_body_part_j_joint_2
                    L2 = np.sqrt((body_part_j_raw[1,0] - Bio_body_parts[j,0,1])**2 +
                                 (body_part_j_raw[1,1] - Bio_body_parts[j,1,1])**2 + 
                                 (body_part_j_raw[1,2] - Bio_body_parts[j,2,1])**2)
            
                    if L1 > L2:
                        if joint_adjust_ID[int(Bio_body_parts[j,3,0])] == 0: # Checks if joint 1 has been adjusted yet (if value is 0 then it has not been adjusted)
                            # Unit vector from joint 2 to 1 for raw body part j
                            w = (1/raw_length_j)*(np.array([body_part_j_raw[0,0],body_part_j_raw[0,1],body_part_j_raw[0,2]]) - 
                                 np.array([body_part_j_raw[1,0],body_part_j_raw[1,1],body_part_j_raw[1,2]]))
                            # creating temp variable to store joint 1 of raw body j
                            temp = body_part_j_raw[1,0:3]
                            # Updating value of joint 1
                            body_part_j_raw[0,0:3] = Bio_lengths[j]*w + temp
                            body_parts_raw_i[j,0,0:3] = body_part_j_raw[0,0:3]
                            Bio_body_parts[j,3,0] = 1 # Joint 1 of body-part j is adjusted and ID updated 

                        # Checks if joint 1 is adjsted and joint 2 is not adjusted
                        if joint_adjust_ID[int(Bio_body_parts[j,3,0])] == 1 and joint_adjust_ID[int(Bio_body_parts[j,3,1])] == 0: 
                            # Unit vector from joint 1 to 2 for raw body part j
                            w = (1/raw_length_j)*(np.array([body_part_j_raw[1,0],body_part_j_raw[1,1],body_part_j_raw[1,2]]) - 
                                np.array([body_part_j_raw[0,0],body_part_j_raw[0,1],body_part_j_raw[0,2]]))
                            # creating temp variable to store joint 1 of raw body j
                            temp = body_part_j_raw[0,0:3]
                            # Updating value of joint 1
                            body_part_j_raw[1,0:3] = Bio_lengths[j]*w + temp
                            body_parts_raw_i[j,1,0:3] = body_part_j_raw[1,0:3]
                            Bio_body_parts[j,3,1] = 1 # Joint 2 of body-part j is adjusted and ID updated 
              
                        # Checks if joint 1 is adjsted and joint 2 is adjusted
                        if joint_adjust_ID[int(Bio_body_parts[j,3,0])] == 1 and joint_adjust_ID[int(Bio_body_parts[j,3,1])] == 1:
                            continue


                    elif L2 > L1:
                        if joint_adjust_ID[int(Bio_body_parts[j,3,1])] == 0: # Checks if joint 2 has been adjusted yet (if value is 0 then it has not been adjusted)
                            # Unit vector from joint 1 to 2 for raw body part j
                            w = (1/raw_length_j)*(np.array([body_part_j_raw[1,0],body_part_j_raw[1,1],body_part_j_raw[1,2]]) - 
                                np.array([body_part_j_raw[0,0],body_part_j_raw[0,1],body_part_j_raw[0,2]]))
                            # creating temp variable to store joint 1 of raw body j
                            temp = body_part_j_raw[0,0:3]
                            # Updating value of joint 1
                            body_part_j_raw[1,0:3] = Bio_lengths[j]*w + temp
                            body_parts_raw_i[j,1,0:3] = body_part_j_raw[1,0:3]
                            Bio_body_parts[j,3,1] = 1 # Joint 2 of body-part j is adjusted and ID updated 
                      
                        # Checks if joint 2 is adjsted and joint 1 is not adjusted
                        if joint_adjust_ID[int(Bio_body_parts[j,3,1])] == 1 and joint_adjust_ID[int(Bio_body_parts[j,3,0])] == 0: # Checks if joint has been adjusted yet (if value is 1 then it has been adjusted and hence we skip to next body part)
                            # Unit vector from joint 2 to 1 for raw body part j
                            w = (1/raw_length_j)*(np.array([body_part_j_raw[0,0],body_part_j_raw[0,1],body_part_j_raw[0,2]]) - 
                                np.array([body_part_j_raw[1,0],body_part_j_raw[1,1],body_part_j_raw[1,2]]))
                            # creating temp variable to store joint 1 of raw body j
                            temp = body_part_j_raw[1,0:3]
                            # Updating value of joint 1
                            body_part_j_raw[0,0:3] = Bio_lengths[j]*w + temp
                            body_parts_raw_i[j,0,0:3] = body_part_j_raw[0,0:3]
                            Bio_body_parts[j,3,0] = 1 # Joint 1 of body-part j is adjusted and ID updated 

                        # Checks if joint 1 is adjsted and joint 2 is adjusted
                        if joint_adjust_ID[int(Bio_body_parts[j,3,0])] == 1 and joint_adjust_ID[int(Bio_body_parts[j,3,1])] == 1:
                            continue
          

            elif i > 0: # Check if the raw-skeleton frame being evaluated is not first frame
                # Obtain body-part j from list of raw-skeleton body parts in array 'body_parts_raw'
                body_part_j_raw = body_parts_raw_i[j,:,:] 
                  
                # all body-part joints for previous frame of raw skeleton
                body_parts_raw_prev_i = np.array([[[skel_body[(i-1),3,2],skel_body[(i-1),3,0],skel_body[(i-1),3,1]],[skel_body[(i-1),2,2],skel_body[(i-1),2,0],skel_body[(i-1),2,1]]],
                                        [[skel_body[(i-1),2,2],skel_body[(i-1),2,0],skel_body[(i-1),2,1]],[skel_body[(i-1),20,2],skel_body[(i-1),20,0],skel_body[(i-1),20,1]]],
                                        [[skel_body[(i-1),20,2],skel_body[(i-1),20,0],skel_body[(i-1),20,1]],[skel_body[(i-1),8,2],skel_body[(i-1),8,0],skel_body[(i-1),8,1]]],
                                        [[skel_body[(i-1),8,2],skel_body[(i-1),8,0],skel_body[(i-1),8,1]],[skel_body[(i-1),9,2],skel_body[(i-1),9,0],skel_body[(i-1),9,1]]],
                                        [[skel_body[(i-1),9,2],skel_body[(i-1),9,0],skel_body[(i-1),9,1]],[skel_body[(i-1),10,2],skel_body[(i-1),10,0],skel_body[(i-1),10,1]]],
                                        [[skel_body[(i-1),20,2],skel_body[(i-1),20,0],skel_body[(i-1),20,1]],[skel_body[(i-1),4,2],skel_body[(i-1),4,0],skel_body[(i-1),4,1]]],
                                        [[skel_body[(i-1),4,2],skel_body[(i-1),4,0],skel_body[(i-1),4,1]],[skel_body[(i-1),5,2],skel_body[(i-1),5,0],skel_body[(i-1),5,1]]],
                                        [[skel_body[(i-1),5,2],skel_body[(i-1),5,0],skel_body[(i-1),5,1]],[skel_body[(i-1),6,2],skel_body[(i-1),6,0],skel_body[(i-1),6,1]]],
                                        [[skel_body[(i-1),20,2],skel_body[(i-1),20,0],skel_body[(i-1),20,1]],[skel_body[(i-1),1,2],skel_body[(i-1),1,0],skel_body[(i-1),1,1]]],
                                        [[skel_body[(i-1),1,2],skel_body[(i-1),1,0],skel_body[(i-1),1,1]],[skel_body[(i-1),0,2],skel_body[(i-1),0,0],skel_body[(i-1),0,1]]],
                                        [[skel_body[(i-1),0,2],skel_body[(i-1),0,0],skel_body[(i-1),0,1]],[skel_body[(i-1),16,2],skel_body[(i-1),16,0],skel_body[(i-1),16,1]]],
                                        [[skel_body[(i-1),0,2],skel_body[(i-1),0,0],skel_body[(i-1),0,1]],[skel_body[(i-1),12,2],skel_body[(i-1),12,0],skel_body[(i-1),12,1]]],
                                        [[skel_body[(i-1),16,2],skel_body[(i-1),16,0],skel_body[(i-1),16,1]],[skel_body[(i-1),17,2],skel_body[(i-1),17,0],skel_body[(i-1),17,1]]],
                                        [[skel_body[(i-1),17,2],skel_body[(i-1),17,0],skel_body[(i-1),17,1]],[skel_body[(i-1),18,2],skel_body[(i-1),18,0],skel_body[(i-1),18,1]]],
                                        [[skel_body[(i-1),18,2],skel_body[(i-1),18,0],skel_body[(i-1),18,1]],[skel_body[(i-1),19,2],skel_body[(i-1),19,0],skel_body[(i-1),19,1]]],
                                        [[skel_body[(i-1),12,2],skel_body[(i-1),12,0],skel_body[(i-1),12,1]],[skel_body[(i-1),13,2],skel_body[(i-1),13,0],skel_body[(i-1),13,1]]],
                                        [[skel_body[(i-1),13,2],skel_body[(i-1),13,0],skel_body[(i-1),13,1]],[skel_body[(i-1),14,2],skel_body[(i-1),14,0],skel_body[(i-1),14,1]]],
                                        [[skel_body[(i-1),14,2],skel_body[(i-1),14,0],skel_body[(i-1),14,1]],[skel_body[(i-1),15,2],skel_body[(i-1),15,0],skel_body[(i-1),15,1]]]])
                      
                # Obtain body-part j of previous frame from list of raw-skeleton body parts in array 'body_parts_raw_prev_i'
                body_part_j_raw_prev_i = body_parts_raw_prev_i[j,:,:] 

                ### Implementing body-part length constraint algorithm:
                # Check length of body part j
                raw_length_j = np.sqrt((body_part_j_raw[0,0] - body_part_j_raw[1,0])**2 +
                                       (body_part_j_raw[0,1] - body_part_j_raw[1,1])**2 + 
                                       (body_part_j_raw[0,2] - body_part_j_raw[1,2])**2)
                if raw_length_j == Bio_lengths[j]:
                    continue
                elif raw_length_j != Bio_lengths[j]:
                    # Euclidean distance between raw_body_part_j_joint_1 and raw_body_part_j_raw_prev_i_joint_1
                    L1 = np.sqrt((body_part_j_raw[0,0] - body_part_j_raw_prev_i[0,0])**2 +
                                 (body_part_j_raw[0,1] - body_part_j_raw_prev_i[0,1])**2 + 
                                 (body_part_j_raw[0,2] - body_part_j_raw_prev_i[0,2])**2)
                    # Euclidean distance between raw_body_part_j_joint_2 and raw_body_part_j_raw_prev_i_joint_2
                    L2 = np.sqrt((body_part_j_raw[1,0] - body_part_j_raw_prev_i[1,0])**2 +
                                 (body_part_j_raw[1,1] - body_part_j_raw_prev_i[1,1])**2 + 
                                 (body_part_j_raw[1,2] - body_part_j_raw_prev_i[1,2])**2)
                    
                    if L1 > L2:
                        if joint_adjust_ID[int(body_part_j_raw[0,3])] == 0: # Checks if joint 1 has been adjusted yet (if value is 0 then it has not been adjusted)
                            # Unit vector from joint 2 to 1 for raw body part j
                            w = (1/raw_length_j)*(np.array([body_part_j_raw[0,0],body_part_j_raw[0,1],body_part_j_raw[0,2]]) - 
                                np.array([body_part_j_raw[1,0],body_part_j_raw[1,1],body_part_j_raw[1,2]]))
                            # creating temp variable to store joint 1 of raw body j
                            temp = body_part_j_raw[1,0:3]
                            # Updating value of joint 1
                            body_part_j_raw[0,0:3] = Bio_lengths[j]*w + temp
                            body_parts_raw_i[j,0,0:3] = body_part_j_raw[0,0:3]
                            joint_adjust_ID[int(body_part_j_raw[0,3])] = 1
                      
                        # Checks if joint 1 is adjsted and joint 2 is not adjusted
                        if joint_adjust_ID[int(body_part_j_raw[0,3])] == 1 and joint_adjust_ID[int(body_part_j_raw[1,3])] == 0: 
                            # Unit vector from joint 1 to 2 for raw body part j
                            w = (1/raw_length_j)*(np.array([body_part_j_raw[1,0],body_part_j_raw[1,1],body_part_j_raw[1,2]]) - 
                                np.array([body_part_j_raw[0,0],body_part_j_raw[0,1],body_part_j_raw[0,2]]))
                            # creating temp variable to store joint 2 of raw body j
                            temp = body_part_j_raw[0,0:3]
                            # Updating value of joint 2
                            body_part_j_raw[1,0:3] = Bio_lengths[j]*w + temp
                            body_parts_raw_i[j,1,0:3] = body_part_j_raw[1,0:3]
                            joint_adjust_ID[int(body_part_j_raw[1,3])] = 1
                      
                        # Checks if joint 1 is adjsted and joint 2 is adjusted
                        if joint_adjust_ID[int(body_part_j_raw[0,3])] == 1 and joint_adjust_ID[int(body_part_j_raw[1,3])] == 1:
                            continue 


                    if L2 > L1:
                        if joint_adjust_ID[int(body_part_j_raw[1,3])] == 0: # Checks if joint 2 has been adjusted yet (if value is 0 then it has not been adjusted)
                            # Unit vector from joint 1 to 2 for raw body part j
                            w = (1/raw_length_j)*(np.array([body_part_j_raw[1,0],body_part_j_raw[1,1],body_part_j_raw[1,2]]) - 
                                np.array([body_part_j_raw[0,0],body_part_j_raw[0,1],body_part_j_raw[0,2]]))
                            # creating temp variable to store joint 2 of raw body j
                            temp = body_part_j_raw[0,0:3]
                            # Updating value of joint 2
                            body_part_j_raw[1,0:3] = Bio_lengths[j]*w + temp
                            body_parts_raw_i[j,1,0:3] = body_part_j_raw[1,0:3]
                            joint_adjust_ID[int(body_part_j_raw[1,3])] = 1

                        # Checks if joint 2 is adjsted and joint 1 is not adjusted
                        if joint_adjust_ID[int(body_part_j_raw[1,3])] == 1 and joint_adjust_ID[int(body_part_j_raw[0,3])] == 0:
                            # Unit vector from joint 2 to 1 for raw body part j
                            w = (1/raw_length_j)*(np.array([body_part_j_raw[0,0],body_part_j_raw[0,1],body_part_j_raw[0,2]]) - 
                                np.array([body_part_j_raw[1,0],body_part_j_raw[1,1],body_part_j_raw[1,2]]))
                            # creating temp variable to store joint 1 of raw body j
                            temp = body_part_j_raw[1,0:3]
                            # Updating value of joint 1
                            body_part_j_raw[0,0:3] = Bio_lengths[j]*w + temp
                            body_parts_raw_i[j,0,0:3] = body_part_j_raw[0,0:3]
                            joint_adjust_ID[int(body_part_j_raw[0,3])] = 1

                        # Checks if joint 1 is adjsted and joint 2 is adjusted
                        if joint_adjust_ID[int(body_part_j_raw[0,3])] == 1 and joint_adjust_ID[int(body_part_j_raw[1,3])] == 1:
                            continue
        
      
      
        ### Updating joints in raw_skeleton joint array for frame i
        # raw skeleton joint 1 update
        skel_body[i,0,2] = body_parts_raw_i[9,1,0] #
        skel_body[i,0,0] = body_parts_raw_i[9,1,1] #
        skel_body[i,0,1] = body_parts_raw_i[9,1,2] #
          
        skel_body[i,1,2] = body_parts_raw_i[8,1,0] #
        skel_body[i,1,0] = body_parts_raw_i[8,1,1] #
        skel_body[i,1,1] = body_parts_raw_i[8,1,2] #

        skel_body[i,2,2] = body_parts_raw_i[0,1,0] #
        skel_body[i,2,0] = body_parts_raw_i[0,1,1] #
        skel_body[i,2,1] = body_parts_raw_i[0,1,2] #

        skel_body[i,3,2] = body_parts_raw_i[0,0,0] #
        skel_body[i,3,0] = body_parts_raw_i[0,0,1] #
        skel_body[i,3,1] = body_parts_raw_i[0,0,2] #
           
        skel_body[i,4,2] = body_parts_raw_i[5,1,0] #
        skel_body[i,4,0] = body_parts_raw_i[5,1,1] #
        skel_body[i,4,1] = body_parts_raw_i[5,1,2] #

        skel_body[i,5,2] = body_parts_raw_i[6,1,0] #
        skel_body[i,5,0] = body_parts_raw_i[6,1,1] #
        skel_body[i,5,1] = body_parts_raw_i[6,1,2] #

        skel_body[i,6,2] = body_parts_raw_i[7,1,0] #
        skel_body[i,6,0] = body_parts_raw_i[7,1,1] #
        skel_body[i,6,1] = body_parts_raw_i[7,1,2] #

        skel_body[i,8,2] = body_parts_raw_i[2,1,0] #
        skel_body[i,8,0] = body_parts_raw_i[2,1,1] #
        skel_body[i,8,1] = body_parts_raw_i[2,1,2] #

        skel_body[i,9,2] = body_parts_raw_i[3,1,0] #
        skel_body[i,9,0] = body_parts_raw_i[3,1,1] #
        skel_body[i,9,1] = body_parts_raw_i[3,1,2] #

        skel_body[i,10,2] = body_parts_raw_i[4,1,0] #
        skel_body[i,10,0] = body_parts_raw_i[4,1,1] #
        skel_body[i,10,1] = body_parts_raw_i[4,1,2] #

        skel_body[i,12,2] = body_parts_raw_i[11,1,0] #
        skel_body[i,12,0] = body_parts_raw_i[11,1,1] #
        skel_body[i,12,1] = body_parts_raw_i[11,1,2] #

        skel_body[i,13,2] = body_parts_raw_i[15,1,0] #
        skel_body[i,13,0] = body_parts_raw_i[15,1,1] #
        skel_body[i,13,1] = body_parts_raw_i[15,1,2] #

        skel_body[i,14,2] = body_parts_raw_i[16,1,0] #
        skel_body[i,14,0] = body_parts_raw_i[16,1,1] #
        skel_body[i,14,1] = body_parts_raw_i[16,1,2] #
         
        skel_body[i,15,2] = body_parts_raw_i[17,1,0] #
        skel_body[i,15,0] = body_parts_raw_i[17,1,1] #
        skel_body[i,15,1] = body_parts_raw_i[17,1,2] #

        skel_body[i,16,2] = body_parts_raw_i[10,1,0] #
        skel_body[i,16,0] = body_parts_raw_i[10,1,1] #
        skel_body[i,16,1] = body_parts_raw_i[10,1,2] #

        skel_body[i,17,2] = body_parts_raw_i[12,1,0] #
        skel_body[i,17,0] = body_parts_raw_i[12,1,1] #
        skel_body[i,17,1] = body_parts_raw_i[12,1,2] #

        skel_body[i,18,2] = body_parts_raw_i[13,1,0] #
        skel_body[i,18,0] = body_parts_raw_i[13,1,1] #
        skel_body[i,18,1] = body_parts_raw_i[13,1,2] #

        skel_body[i,19,2] = body_parts_raw_i[14,1,0] #
        skel_body[i,19,0] = body_parts_raw_i[14,1,1] #
        skel_body[i,19,1] = body_parts_raw_i[14,1,2] #

        skel_body[i,20,2] = body_parts_raw_i[1,1,0] #
        skel_body[i,20,0] = body_parts_raw_i[1,1,1] #
        skel_body[i,20,1] = body_parts_raw_i[1,1,2] #

    return skel_body
          
