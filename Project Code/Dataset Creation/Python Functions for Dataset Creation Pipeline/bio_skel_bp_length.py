#### THIS FUNCTION RETURNS THE LENGTHS OF ALL BODY PARTS OF THE BIO-CONSTRAINED SKELETON AS A LIST CALLED 'BIO_LENGTHS'
# Importing Libraries
import numpy as np

def bio_skel_bp_length(skel_body):
    ### Determining Lengths of Bio-Constrained Skeleton Frame

    # skel_body is the 1st frame of the ith raw skeleton action array of size
    # (f,j,c), where f = number of frames in action, j = number of joints
    # in skeleton and c = dimensions of coordinates of each joint (i.e. c = 3)
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

    skel_1_body = skel_body[:,:,:] # Joint coordinates of first frame of raw skeleton

    frame = 0 # frame number (first frame of raw_skeleton data)

    # Raw body part 4 (11th and 10th joints in raw_skeleton) = Bio_Skeleton body part 5 (5th and 6th joints of Bio-skeleton):
    x_b4_1 = skel_1_body[frame ,10,2] 
    x_b4_2 = skel_1_body[frame ,9,2] 
    y_b4_1 = skel_1_body[frame ,10,0]
    y_b4_2 = skel_1_body[frame ,9,0] 
    z_b4_1 = skel_1_body[frame ,10,1]
    z_b4_2 = skel_1_body[frame ,9,1] 

    x_b4 = np.array([x_b4_1, x_b4_2])
    y_b4 = np.array([y_b4_1, y_b4_2])
    z_b4 = np.array([z_b4_1, z_b4_2])

    L5 = np.sqrt(np.square((x_b4[0] - x_b4[1])) + np.square((y_b4[0] - y_b4[1])) + np.square((z_b4[0] - z_b4[1])))



    # Raw body part 5 (10th and 9th joints in skeleton) = Bio_Skeleton body part 4 (5th and 4th joints of Bio-skeleton):
    x_b5_1 = skel_1_body[frame ,9,2] 
    x_b5_2 = skel_1_body[frame ,8,2] 
    y_b5_1 = skel_1_body[frame ,9,0]
    y_b5_2 = skel_1_body[frame ,8,0] 
    z_b5_1 = skel_1_body[frame ,9,1]
    z_b5_2 = skel_1_body[frame ,8,1] 

    x_b5 = np.array([x_b5_1, x_b5_2])
    y_b5 = np.array([y_b5_1, y_b5_2])
    z_b5 = np.array([z_b5_1, z_b5_2])

    L4 = np.sqrt(np.square((x_b5[0] - x_b5[1])) + np.square((y_b5[0] - y_b5[1])) + np.square((z_b5[0] - z_b5[1])))



    # Raw body part 6 (9th and 21th joints in skeleton) = Bio_Skeleton body part 3 (4th and 20th joints of Bio-skeleton):
    x_b6_1 = skel_1_body[frame ,8,2] 
    x_b6_2 = skel_1_body[frame ,20,2] 
    y_b6_1 = skel_1_body[frame ,8,0]
    y_b6_2 = skel_1_body[frame ,20,0] 
    z_b6_1 = skel_1_body[frame ,8,1]
    z_b6_2 = skel_1_body[frame ,20,1] 

    x_b6 = np.array([x_b6_1, x_b6_2])
    y_b6 = np.array([y_b6_1, y_b6_2])
    z_b6 = np.array([z_b6_1, z_b6_2])

    L3 = np.sqrt(np.square((x_b6[0] - x_b6[1])) + np.square((y_b6[0] - y_b6[1])) + np.square((z_b6[0] - z_b6[1])))



    # Raw body part 7 (4th and 3th joints in skeleton) = Bio_Skeleton body part 1 (3rd and 2nd joints of Bio-skeleton):
    x_b7_1 = skel_1_body[frame ,3,2] 
    x_b7_2 = skel_1_body[frame ,2,2] 
    y_b7_1 = skel_1_body[frame ,3,0]
    y_b7_2 = skel_1_body[frame ,2,0] 
    z_b7_1 = skel_1_body[frame ,3,1]
    z_b7_2 = skel_1_body[frame ,2,1] 

    x_b7 = np.array([x_b7_1, x_b7_2])
    y_b7 = np.array([y_b7_1, y_b7_2])
    z_b7 = np.array([z_b7_1, z_b7_2])

    L1 = np.sqrt(np.square((x_b7[0] - x_b7[1])) + np.square((y_b7[0] - y_b7[1])) + np.square((z_b7[0] - z_b7[1])))



    # Raw body part 8 (3th and 21th joints in skeleton) = = Bio_Skeleton body part 2 (2nd and 20th joints of Bio-skeleton):
    x_b8_1 = skel_1_body[frame ,2,2] 
    x_b8_2 = skel_1_body[frame ,20,2] 
    y_b8_1 = skel_1_body[frame ,2,0]
    y_b8_2 = skel_1_body[frame ,20,0] 
    z_b8_1 = skel_1_body[frame ,2,1]
    z_b8_2 = skel_1_body[frame ,20,1] 

    x_b8 = np.array([x_b8_1, x_b8_2])
    y_b8 = np.array([y_b8_1, y_b8_2])
    z_b8 = np.array([z_b8_1, z_b8_2])

    L2 = np.sqrt(np.square((x_b8[0] - x_b8[1])) + np.square((y_b8[0] - y_b8[1])) + np.square((z_b8[0] - z_b8[1])))



    # Raw body part 9 (21th and 5th joints in skeleton) = Bio_Skeleton body part 6 (20th and 8th joints of Bio-skeleton):
    x_b9_1 = skel_1_body[frame ,20,2] 
    x_b9_2 = skel_1_body[frame ,4,2] 
    y_b9_1 = skel_1_body[frame ,20,0]
    y_b9_2 = skel_1_body[frame ,4,0] 
    z_b9_1 = skel_1_body[frame ,20,1]
    z_b9_2 = skel_1_body[frame ,4,1] 

    x_b9 = np.array([x_b9_1, x_b9_2])
    y_b9 = np.array([y_b9_1, y_b9_2])
    z_b9 = np.array([z_b9_1, z_b9_2])

    L6 = np.sqrt(np.square((x_b9[0] - x_b9[1])) + np.square((y_b9[0] - y_b9[1])) + np.square((z_b9[0] - z_b9[1])))



    # Raw body part 10 (5th and 6th joints in skeleton) = Bio_Skeleton body part 7 (8th and 9th joints of Bio-skeleton):
    x_b10_1 = skel_1_body[frame ,4,2] 
    x_b10_2 = skel_1_body[frame ,5,2] 
    y_b10_1 = skel_1_body[frame ,4,0]
    y_b10_2 = skel_1_body[frame ,5,0] 
    z_b10_1 = skel_1_body[frame ,4,1]
    z_b10_2 = skel_1_body[frame ,5,1] 

    x_b10 = np.array([x_b10_1, x_b10_2])
    y_b10 = np.array([y_b10_1, y_b10_2])
    z_b10 = np.array([z_b10_1, z_b10_2])

    L7 = np.sqrt(np.square((x_b10[0] - x_b10[1])) + np.square((y_b10[0] - y_b10[1])) + np.square((z_b10[0] - z_b10[1])))



    # Raw body part 11 (6th and 7th joints in skeleton) = Bio_Skeleton body part 8 (9th and 10th joints of Bio-skeleton):
    x_b11_1 = skel_1_body[frame ,5,2] 
    x_b11_2 = skel_1_body[frame ,6,2] 
    y_b11_1 = skel_1_body[frame ,5,0]
    y_b11_2 = skel_1_body[frame ,6,0] 
    z_b11_1 = skel_1_body[frame ,5,1]
    z_b11_2 = skel_1_body[frame ,6,1] 

    x_b11 = np.array([x_b11_1, x_b11_2])
    y_b11 = np.array([y_b11_1, y_b11_2])
    z_b11 = np.array([z_b11_1, z_b11_2])

    L8 = np.sqrt(np.square((x_b11[0] - x_b11[1])) + np.square((y_b11[0] - y_b11[1])) + np.square((z_b11[0] - z_b11[1])))



    # Raw body part 15 (21th and 2th joints in skeleton) = Bio_Skeleton body part 9 (20th and 1st joints of Bio-skeleton):
    x_b15_1 = skel_1_body[frame ,20,2] 
    x_b15_2 = skel_1_body[frame ,1,2] 
    y_b15_1 = skel_1_body[frame ,20,0]
    y_b15_2 = skel_1_body[frame ,1,0] 
    z_b15_1 = skel_1_body[frame ,20,1]
    z_b15_2 = skel_1_body[frame ,1,1] 

    x_b15 = np.array([x_b15_1, x_b15_2])
    y_b15 = np.array([y_b15_1, y_b15_2])
    z_b15 = np.array([z_b15_1, z_b15_2])

    L9 = np.sqrt(np.square((x_b15[0] - x_b15[1])) + np.square((y_b15[0] - y_b15[1])) + np.square((z_b15[0] - z_b15[1])))



    # Raw body part 16 (2th and 1th joints in skeleton) = Bio_Skeleton body part 10 (1st and oth joints of Bio-skeleton):
    x_b16_1 = skel_1_body[frame ,1,2] 
    x_b16_2 = skel_1_body[frame ,0,2] 
    y_b16_1 = skel_1_body[frame ,1,0]
    y_b16_2 = skel_1_body[frame ,0,0] 
    z_b16_1 = skel_1_body[frame ,1,1]
    z_b16_2 = skel_1_body[frame ,0,1] 

    x_b16 = np.array([x_b16_1, x_b16_2])
    y_b16 = np.array([y_b16_1, y_b16_2])
    z_b16 = np.array([z_b16_1, z_b16_2])

    L10 = np.sqrt(np.square((x_b16[0] - x_b16[1])) + np.square((y_b16[0] - y_b16[1])) + np.square((z_b16[0] - z_b16[1])))



    # Raw body part 17 (1th and 17th joints in skeleton) = Bio_Skeleton body part 11 (oth and 12th joints of Bio-skeleton):
    x_b17_1 = skel_1_body[frame ,0,2] 
    x_b17_2 = skel_1_body[frame ,16,2] 
    y_b17_1 = skel_1_body[frame ,0,0]
    y_b17_2 = skel_1_body[frame ,16,0] 
    z_b17_1 = skel_1_body[frame ,0,1]
    z_b17_2 = skel_1_body[frame ,16,1] 

    x_b17 = np.array([x_b17_1, x_b17_2])
    y_b17 = np.array([y_b17_1, y_b17_2])
    z_b17 = np.array([z_b17_1, z_b17_2])

    L11 = np.sqrt(np.square((x_b17[0] - x_b17[1])) + np.square((y_b17[0] - y_b5[1])) + np.square((z_b17[0] - z_b17[1])))



    # Raw body part 18 (1th and 13th joints in skeleton) = Bio_Skeleton body part 12 (oth and 16th joints of Bio-skeleton):
    x_b18_1 = skel_1_body[frame ,0,2] 
    x_b18_2 = skel_1_body[frame ,12,2] 
    y_b18_1 = skel_1_body[frame ,0,0]
    y_b18_2 = skel_1_body[frame ,12,0] 
    z_b18_1 = skel_1_body[frame ,0,1]
    z_b18_2 = skel_1_body[frame ,12,1] 

    x_b18 = np.array([x_b18_1, x_b18_2])
    y_b18 = np.array([y_b18_1, y_b18_2])
    z_b18 = np.array([z_b18_1, z_b18_2])

    L12 = np.sqrt(np.square((x_b18[0] - x_b18[1])) + np.square((y_b18[0] - y_b18[1])) + np.square((z_b18[0] - z_b18[1])))



    # Raw body part 19 (17th and 18th joints in skeleton) = Bio_Skeleton body part 13 (12th and 13th joints of Bio-skeleton):
    x_b19_1 = skel_1_body[frame ,16,2] 
    x_b19_2 = skel_1_body[frame ,17,2] 
    y_b19_1 = skel_1_body[frame ,16,0]
    y_b19_2 = skel_1_body[frame ,17,0] 
    z_b19_1 = skel_1_body[frame ,16,1]
    z_b19_2 = skel_1_body[frame ,17,1] 

    x_b19 = np.array([x_b19_1, x_b19_2])
    y_b19 = np.array([y_b19_1, y_b19_2])
    z_b19 = np.array([z_b19_1, z_b19_2])

    L13 = np.sqrt(np.square((x_b19[0] - x_b19[1])) + np.square((y_b19[0] - y_b19[1])) + np.square((z_b19[0] - z_b19[1])))



    # Raw body part 20 (13th and 14th joints in skeleton) = Bio_Skeleton body part 16 (16th and 17th joints of Bio-skeleton):
    x_b20_1 = skel_1_body[frame ,12,2] 
    x_b20_2 = skel_1_body[frame ,13,2] 
    y_b20_1 = skel_1_body[frame ,12,0]
    y_b20_2 = skel_1_body[frame ,13,0] 
    z_b20_1 = skel_1_body[frame ,12,1]
    z_b20_2 = skel_1_body[frame ,13,1] 

    x_b20 = np.array([x_b20_1, x_b20_2])
    y_b20 = np.array([y_b20_1, y_b20_2])
    z_b20 = np.array([z_b20_1, z_b20_2])

    L16 = np.sqrt(np.square((x_b20[0] - x_b20[1])) + np.square((y_b20[0] - y_b20[1])) + np.square((z_b20[0] - z_b20[1])))



    # Raw body part 21 (18th and 19th joints in skeleton) = Bio_Skeleton body part 14 (13th and 14th joints of Bio-skeleton):
    x_b21_1 = skel_1_body[frame ,17,2] 
    x_b21_2 = skel_1_body[frame ,18,2] 
    y_b21_1 = skel_1_body[frame ,17,0]
    y_b21_2 = skel_1_body[frame ,18,0] 
    z_b21_1 = skel_1_body[frame ,17,1]
    z_b21_2 = skel_1_body[frame ,18,1] 

    x_b21 = np.array([x_b21_1, x_b21_2])
    y_b21 = np.array([y_b21_1, y_b21_2])
    z_b21 = np.array([z_b21_1, z_b21_2])

    L14 = np.sqrt(np.square((x_b21[0] - x_b21[1])) + np.square((y_b21[0] - y_b21[1])) + np.square((z_b21[0] - z_b21[1])))



    # Raw body part 22 (14th and 15th joints in skeleton) = Bio_Skeleton body part 17 (17th and 18th joints of Bio-skeleton):
    x_b22_1 = skel_1_body[frame ,13,2] 
    x_b22_2 = skel_1_body[frame ,14,2] 
    y_b22_1 = skel_1_body[frame ,13,0]
    y_b22_2 = skel_1_body[frame ,14,0] 
    z_b22_1 = skel_1_body[frame ,13,1]
    z_b22_2 = skel_1_body[frame ,14,1] 

    x_b22 = np.array([x_b22_1, x_b22_2])
    y_b22 = np.array([y_b22_1, y_b22_2])
    z_b22 = np.array([z_b22_1, z_b22_2])

    L17 = np.sqrt(np.square((x_b22[0] - x_b22[1])) + np.square((y_b22[0] - y_b22[1])) + np.square((z_b22[0] - z_b22[1])))



    # Raw body part 23 (19th and 20th joints in skeleton) = Bio_Skeleton body part 15 (14th and 15th joints of Bio-skeleton):
    x_b23_1 = skel_1_body[frame ,18,2] 
    x_b23_2 = skel_1_body[frame ,19,2] 
    y_b23_1 = skel_1_body[frame ,18,0]
    y_b23_2 = skel_1_body[frame ,19,0] 
    z_b23_1 = skel_1_body[frame ,18,1]
    z_b23_2 = skel_1_body[frame ,19,1] 

    x_b23 = np.array([x_b23_1, x_b23_2])
    y_b23 = np.array([y_b23_1, y_b23_2])
    z_b23 = np.array([z_b23_1, z_b23_2])

    L15 = np.sqrt(np.square((x_b23[0] - x_b23[1])) + np.square((y_b23[0] - y_b23[1])) + np.square((z_b23[0] - z_b23[1])))



    # Raw body part 24 (15th and 16th joints in skeleton) = Bio_Skeleton body part 18 (18th and 19th joints of Bio-skeleton):
    x_b24_1 = skel_1_body[frame ,14,2] 
    x_b24_2 = skel_1_body[frame ,15,2] 
    y_b24_1 = skel_1_body[frame ,14,0]
    y_b24_2 = skel_1_body[frame ,15,0] 
    z_b24_1 = skel_1_body[frame ,14,1]
    z_b24_2 = skel_1_body[frame ,15,1] 

    x_b24 = np.array([x_b24_1, x_b24_2])
    y_b24 = np.array([y_b24_1, y_b24_2])
    z_b24 = np.array([z_b24_1, z_b24_2])

    L18 = np.sqrt(np.square((x_b24[0] - x_b24[1])) + np.square((y_b24[0] - y_b24[1])) + np.square((z_b24[0] - z_b24[1])))

    # Averaging mirroring body parts so that pairs of body parts have same length (i.e. both right and left arm have same length)
    # L3 and L6 are paired
    L3 = np.mean([L3,L6])
    L6 = L3
    # L4 and L7 are paired
    L4 = np.mean([L4,L7])
    L7 = L4
    # L5 and L8 are paired
    L5 = np.mean([L5,L8])
    L8 = L5
    # L11 and L12 are paired
    L11 = np.mean([L11,L12])
    L12 = L11
    # L13 and L16 are paired
    L13 = np.mean([L13,L16])
    L16 = L13
    # L14 and L17 are paired
    L14 = np.mean([L14,L17])
    L17 = L14
    # L15 and L18 are paired
    L15 = np.mean([L15,L18])
    L18 = L15


    ### Setting up body part lengths (Bio-Skeleton):
    """ There are 18 body parts in the Bio-Skeleton. All body-parts except body-parts 15 and 18 have a length of 1, 
    Body-parts 15 and 18 have a length of 0.2 as they represent the feet of the skeleton. Joints for each body-part are 
    shown in section under comment ### Visualising Bio-Skeleton:"""
    Bio_lengths = [L1,L2,L3,L4,L5,L6,L7,L8,L9,L10,L11,L12,L13,L14,L15,L16,L17,L18]

    return Bio_lengths
