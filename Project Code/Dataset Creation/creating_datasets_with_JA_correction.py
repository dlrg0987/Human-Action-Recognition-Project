#### CURRENT ERROR THAT IS APPEARING (SOLVED): numpy.linalg.LinAlgError: SVD did not converge
#### SOLUTION TO THAT ERROR FROM STACK OVERFLOW: https://stackoverflow.com/questions/21827594/raise-linalgerrorsvd-did-not-converge-linalgerror-svd-did-not-converge-in-m
#### SOLUTION (ACTUAL):https://stackoverflow.com/questions/52458409/how-to-remove-nan-and-inf-values-from-a-numpy-matrix

#### NEWTEST ERROR: numpy.core._exceptions._ArrayMemoryError: Unable to allocate 1.15 MiB for an array with shape (224, 224, 3) and data type float64
#### SOLUTION: try to extract the 10 classes one at a time to reduce amount of RAM/memeory used 

import os
import numpy as np
from skimage.transform import resize
import matplotlib.image

# importing all functions from python files that constitute the pre-processing pipeline
from bio_rotate import *
from bio_rotate_ja import *
from bio_skel_bp_length import *
from bio_skel_joint_create import *
from bone_length_constraint import *
from JEA_create import *
from JEDM_create import *
from joint_angle_correction import *
from translate_bio_skel import *

#skeleton_np_arrays = np.array([]) # creating empty numpy array to hold all skeletons
#num_p_IDs = 0


### CROSS SUBEJCT PROTOCOL ID NUMBERS ###
# Person ID numbers for training set in Cross-Subject Protocol
person_ID_list_CS_TRS = ['01','02','04','05','08','09','13','14','15','16','17','18','19','25','27','28','31','34','35','38']
# Person ID numbers for cross validation and testing set in Cross-Subject Protocol
person_ID_list_CS_CVTES = ['03','06','07','10','11','12','20','21','22','23','24','26','29','30','32','33','36','37','39','40']
# Setup ID numbers for cross validation set in Cross-Subject Protocol
Setup_ID_list_CS_CVS = ['02','04','06','08','10','12','14','16']
# Setup ID numbers for testing set in Cross-Subject Protocol
Setup_ID_list_CS_TES = ['01','03','05','07','09','11','13','15','17']

### CROSS VIEW PROTOCOL ID NUMBERS ###
# Camera ID numbers for training set in Cross-View Protocol
camera_ID_list_CV_TRS = ['01','02']
# Camera ID numbers for cross validation and testing set in Cross-View Protocol
camera_ID_list_CV_CVTES = ['03']
# Setup ID numbers for cross validation set in Cross-View Protocol
Setup_ID_list_CV_CVS = ['02','04','06','08','10','12','14','16']
# Setup ID numbers for testing set in Cross-View Protocol
Setup_ID_list_CV_TES = ['01','03','05','07','09','11','13','15','17']


#file path of folder containing all numpy arrays of raw skeletons
load_file_path = 'C:/Users/moses/Documents/ES327 - Individual Project (Engineering Year 3)/Python code to convert raw skeleton data to numpy arrays/All files + code to convert raw skeletons to numpy arrays/Numpy arrays of skeleton data'

#file path of folder where skeleton images will be stored as a dataset

for array in os.walk(load_file_path):
    skeleton_list = array[2] # extract skeleton list containing npy file names as srings for all raw skeletons from folder

#print(len(skeleton_list)) # value should be 56000 ish
    
# Extract only class in variable 'C'
C = 59

for i in range(len(skeleton_list)):
    skel_dict_i_path = load_file_path + '/' + skeleton_list[i] #file path of ith skeleton dictionary
    skel_dict_i = np.load(skel_dict_i_path,allow_pickle=True).item() #ith skeleton dictionary containing ith raw skeleton array

    skel_keys = list(skel_dict_i.keys())
    for key in range(len(skel_keys)):
        if skel_keys[key] == 'skel_body' or skel_keys[key] == 'skel_body0':
            skel_array_i = skel_dict_i.get(skel_keys[key])
            break
        else:
            continue
        
    skel_file_name = skel_dict_i.get('file_name')
    #print(skel_file_name)


    setup_ID = skel_file_name[2:4]
    camera_ID = skel_file_name[6:8]
    person_ID = skel_file_name[10:12]
    class_ID = skel_file_name[18:20]

    # only extract first 10 classes
    # extract following classes:
    # class 1 = drinking water (extracted fully)
    # class 6 = picking something up (extracted fully)
    # class 7 = throw (extracted fully)
    # class 8 = sitting down (extracted fully)
    # class 9 = standing up (extracted fully)
    # class 24 = kicking (extracted fully)
    # class 26 = hopping on one leg (extracted fully)
    # class 27 = jumping up (extracted fully)
    # class 43 = falling (extracted fully)
    # class 59 = walking (extracted fully)

    # Extract only class in variable 'C' defined before foor loop above
    
    if (int(class_ID) != C): #and(int(class_ID) != 6)and(int(class_ID) != 7)and(int(class_ID) != 8)and(int(class_ID) != 9)and(int(class_ID) != 24)and(int(class_ID) != 26)and(int(class_ID) != 27)and(int(class_ID) != 43)and(int(class_ID) != 59):
        continue
    
    '''
    # This code extracts all the ID numbers of the subjects/persons from whom skeletons are created in the NTU-RGB-D dataset
    if all(ID != person_ID for ID in person_ID_list):
        person_ID_list.append(person_ID)
        num_p_IDs = num_p_IDs + 1
    '''
    
    '''
    CROSS_SUBJECT PROTOCOL:
    
    person IDs (P0XX, where XX = ID number) includes:
    '01', '02', '03', '04', '05', '06', '07', '08', '09',
    '10', '11', '12', '13', '14', '15', '16', '17', '18',
    '19', '20', '21', '22', '23', '24', '25', '26', '27',
    '28', '29', '30', '31', '32', '33', '34', '35', '36',
    '38', '37', '39', '40' as ID numbers

    Will need to include person IDs '01', '02', '04',
    '05', '08', '09', '13', '14', '15', '16', '17', '18',
    '19', '25', '27', '28', '31', '34', '35', '38' in Training Dataset.

    Will need to include person IDs '03', '06', '07',
    '10', '11', '12', '20', '21', '22', '23', '24', '26', 
    '29', '30', '32', '33', '36', '37', '39', '40' in Validation + Testing Dataset.

    setup IDs (S0xx where XX = ID number) has 17 values:
    - Include even setup IDs in Validation Dataset
    - Include odd setup IDs in Testing Dataset

    CROSS_VIEW PROTOCOL:

    camera IDs (C0XX, where XX = ID number) includes:
    '01', '02', '03'

    Will need to include camera IDs '01' and '02' in Training Dataset

    Will need to include camera ID '03' in Validation and Testing Datasets

    setup IDs (P0xx where XX = ID number) has 17 values:
    - Include even setup IDs in Validation Dataset
    - Include odd setup IDs in Testing Dataset
    
    '''
    
    #print(skel_array_i.shape)
    #print('iteration: ' + str(i))
    
    dataset_CS = ''
    if any(ID == person_ID for ID in person_ID_list_CS_TRS):
        dataset_CS = 'Training Dataset'
    elif any(ID == person_ID for ID in person_ID_list_CS_CVTES) and any(ID == setup_ID for ID in Setup_ID_list_CS_CVS):
        dataset_CS = 'Cross Validation Dataset'
    elif any(ID == person_ID for ID in person_ID_list_CS_CVTES) and any(ID == setup_ID for ID in Setup_ID_list_CS_TES):
        dataset_CS = 'Testing Dataset'
    
    dataset_CV = ''
    if any(ID == camera_ID for ID in camera_ID_list_CV_TRS):
        dataset_CV = 'Training Dataset'
    elif any(ID == camera_ID for ID in camera_ID_list_CV_CVTES) and any(ID == setup_ID for ID in Setup_ID_list_CV_CVS):
        dataset_CV = 'Cross Validation Dataset'
    elif any(ID == camera_ID for ID in camera_ID_list_CV_CVTES) and any(ID == setup_ID for ID in Setup_ID_list_CV_TES):
        dataset_CV = 'Testing Dataset'

    class_folder = 'class_'+class_ID

    save_path_CS_JEDM = 'C:/Users/moses/Documents/ES327 - Individual Project (Engineering Year 3)/Final Datasets/Datasets with joint angle correction/Cross Subject Protocol Datasets (JAC)/'+dataset_CS+'/JEDM images/'+class_folder
    save_path_CS_JEA = 'C:/Users/moses/Documents/ES327 - Individual Project (Engineering Year 3)/Final Datasets/Datasets with joint angle correction/Cross Subject Protocol Datasets (JAC)/'+dataset_CS+'/JEA images/'+class_folder
    save_path_CV_JEDM = 'C:/Users/moses/Documents/ES327 - Individual Project (Engineering Year 3)/Final Datasets/Datasets with joint angle correction/Cross View Protocol Datasets (JAC)/'+dataset_CV+'/JEDM images/'+class_folder
    save_path_CV_JEA = 'C:/Users/moses/Documents/ES327 - Individual Project (Engineering Year 3)/Final Datasets/Datasets with joint angle correction/Cross View Protocol Datasets (JAC)/'+dataset_CV+'/JEA images/'+class_folder 
    
    ##### Implementing full pre-processing pipeline #####

    Bio_lengths = bio_skel_bp_length(skel_array_i)
    Bio_joint = bio_skel_joint_create(Bio_lengths)
    Bio_joint = translate_bio_skel(skel_array_i, Bio_joint)
    Bio_body_parts = bio_rotate(skel_array_i, Bio_joint, 0)
    skel_array_i = bone_length_constraint(skel_array_i, Bio_body_parts, Bio_lengths)
    skel_array_i_new = joint_angle_correction(skel_array_i, Bio_joint)
    JEDM_im_array = JEDM_create(skel_array_i_new)
    JEA_im_array = JEA_create(skel_array_i_new, Bio_joint)
    #JEA_im_array  = np.absolute(JEA_im_array) # making sure negative values are not present 
    #print(np.amax(JEA_im_array))

    
    #### Saving JEA and JEDM numpy arrays as jpeg images in correct directory/path

    matplotlib.image.imsave((save_path_CS_JEDM+('/image_JEDM_CS_'+str(i)+'.jpeg')),JEDM_im_array)
    matplotlib.image.imsave((save_path_CS_JEA+('/image_JEA_CS_'+str(i)+'.jpeg')),JEA_im_array)
    matplotlib.image.imsave((save_path_CV_JEDM+('/image_JEDM_CV_'+str(i)+'.jpeg')),JEDM_im_array)
    matplotlib.image.imsave((save_path_CV_JEA+('/image_JEA_CV_'+str(i)+'.jpeg')),JEA_im_array)

    '''
    if i == 3:
        print('complete')
        break
    '''
    
    #if (i%1000) == 0:
        #print((str(i)+'th iteration'))
    
#print(skeleton_np_arrays.shape) ### index needs changing, it is not 0 but 0,:,:,: I think. print dimensions of skeleton_np.arrays to find out
#print(num_p_IDs)
#print(person_ID_list)
print('Class '+str(C)+' has been extracted fully')
 
    
