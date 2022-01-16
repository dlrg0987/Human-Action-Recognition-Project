#### THIS FUNCTION WILL TRANSLATE THE BIO-CONSTRAINED SKELETON SO THAT JOINT 0 OF BIO_SKEL MATCHES JOINT 0 OF RAW_SKEL (JOINT 0 = MIDDLE OF HIP = FIXED JOINT)

# Importing Libraries
import numpy as np

def translate_bio_skel(skel_body, Bio_joint):
    ### input skel_body and Bio_joint and return translated joints in variable called Bio_joint

    ### Translating Bio-skeleton so that bio-joint 0 has the same coordinate as raw-joint 1 for the first frame of raw skeleton
    """This is done so that the centre of the bio skeleton and the raw skeleton are the same"""
    '''
    # Importing Libraries
    import numpy as np
    '''

    dims = skel_body.shape
    
    # skel_body is the 1st frame of the ith raw skeleton action array of size
    # (f,j,c), where f = number of frames in action, j = number of joints
    # in skeleton and c = dimensions of coordinates of each joint (i.e. c = 3)

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

    raw_skel_joint = np.array([skel_body[0,0,:],skel_body[0,1,:],skel_body[0,2,:],skel_body[0,3,:],
                              skel_body[0,8,:],skel_body[0,9,:],skel_body[0,10,:],skel_body[0,4,:],
                              skel_body[0,5,:], skel_body[0,6,:],skel_body[0,16,:],skel_body[0,17,:],
                              skel_body[0,18,:],skel_body[0,19,:],skel_body[0,12,:],skel_body[0,13,:],
                              skel_body[0,14,:],skel_body[0,15,:],skel_body[0,20,:]])

    # vector from bio-joint 0 to raw-joint 1 = (raw-joint 1 - bio-joint 0)
    trans_vec = skel_body[0,0,:] - Bio_joint[0,:] 
    #print(trans_vec)

    # New position of Bio Skeleton:
    temp = Bio_joint
    Bio_joint = temp + trans_vec

    return Bio_joint
