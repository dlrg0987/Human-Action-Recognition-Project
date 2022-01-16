# importing relevant libraries
import numpy as np
from skimage.transform import resize

def JEDM_create(skel_body):

    # JEDMs feature extraction algorithm (JEDM = Joint Euler Distance Matrix)

    '''
    # importing relevant libraries
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from skimage.transform import resize
    from PIL import Image as im 
    '''

    dims = skel_body.shape # dimensions of numpy array skel_body

    # Creating empty array to store skeleotn image representations that contain spatial and temporal features

    JEDM_image = np.zeros((dims[0],171))
    JEDM = np.zeros((19, 19))
    for i in range(dims[0]):
        skel_frame_i = skel_body[i,:,:] # obtaining ith frame of skeleton
        JEDM_row = np.array([]) # reseting array that holds JEDM values of each frame in a single column

        # obtaining full JEDM matrix
        for j in range(19):
            for k in range(19):
                if (j == 7 or k == 7) or (j == 11 or k == 11) or (j == 21 or k == 21) or (j == 22 or k == 22) or (j == 23 or k == 23) or (j == 24 or k == 24):
                    continue
                else:
                    JEDM[j,k] = np.sqrt((skel_frame_i[j,2] - skel_frame_i[k,2])**2 + (skel_frame_i[j,0] - skel_frame_i[k,0])**2 + (skel_frame_i[j,1] - skel_frame_i[k,1])**2)
      
        # extracting non zero and non repeating values from JEDM (i.e. ||P_j - P_k|| for which j not equal to k and excluding ||P_k - P_j||)
        for rows in range(18): # 18th triangle number = 171 = number of values in JEDM_row
            JEDM_row =  np.append(JEDM_row, JEDM[rows,(rows+1):19])
            #print(JEDM_row.shape)
      
        for index in range(len(JEDM_row) - 1):
            JEDM_image[i,index] = JEDM_row[index]
      



    JEDM_image = np.transpose(JEDM_image)
    #print(JEDM_image.shape)
    #print(JEDM_image[0,:])
    #JEDM_image = np.uint8(JEDM_image)

    JEDM_image_resized = resize(JEDM_image, (224, 224))

    # Normalising JEDM_image so that all values are between 0 - 1
    lam = np.amax(JEDM_image)
    JEDM_image_resized = np.dot((1/lam),JEDM_image_resized)
    
    JEDM_im_array = np.zeros((224,224,3))
    JEDM_im_array[:,:,0] = JEDM_image_resized
    JEDM_im_array[:,:,1] = JEDM_image_resized
    JEDM_im_array[:,:,2] = JEDM_image_resized

    '''
    plt.imshow(JEDM_image_resized)
    plt.title('Image Representation of skeleton JEDMs')
    plt.show()

    # converting numpy array image representation to png image
    matplotlib.image.imsave('test.png', JEDM_image_resized)
    test_JEDM = plt.imread("test.png")
    
    JEDM_image_data = im.fromarray(JEDM_image_resized) 
          
    # saving the final output  
    # as a JPEG file 
    JEDM_image_data.save('test.jpeg') 

    image = Image.open('test.jpeg')
    # convert image to numpy array
    data = asarray(image)
    
    plt.imshow(test_JEDM)
    plt.title('Image Representation of skeleton JEDMs after converted to JPEG')
    plt.show()
    '''

    return JEDM_im_array
