#american signs (a-b) and (1-9)

import mediapipe as mp
import cv2
import os
import matplotlib.pyplot as plt  # Import matplotlib for plotting
import pickle  # pickle libary used to save datasets

#----
# three landmarks that will be useful in terms of drawing over the image(landmark) (drawing the letter)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
#---- 

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# iterate and exactrate the landmarks (then save the data into a file), then take all the landmarks that were detected and create an array 

DATA_DIR = '/Users/crinitinikos/Desktop/summer coding porjects/sign language detector/data'
data = [] # data
labels = [] # for each one of the data poitns 
 
for dir_ in os.listdir(DATA_DIR):
    if not os.path.isdir(os.path.join(DATA_DIR, dir_)):
        continue
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)): # not one image anymore [:1]: # [:1] test
        img = cv2.imread(os.path.join(DATA_DIR,dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # convert image into rgb so it can be converted into mediapipe, mediapipe sees only rgb with the landmark detection

        data_array = [] # where the x and y coordinates will be stored

        x_ = []
        y_ = []

        results = hands.process(img_rgb) #code here and below will iterate
        # iterating becasue there could be 0, 1, 2, 3 hands, you wont know
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
            #creating array
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_array.append(x - min(x_))
                    data_array.append(y - min(y_))

            data.append(data_array) # create any entire list of all the arrrays
            labels.append(dir_)

f = open('/Users/crinitinikos/Desktop/summer coding porjects/sign language detector/data/data.pickle', 'wb') # write as bytes
pickle.dump({'data': data, 'labels': labels}, f)
f.close()
            
            
'''         for i in range(len(hand_landmarks.landmark)):
                    print(hand_landmarks.landmark[i])
                    #for each one of the landmarks (we get back 3 values X, Y, z)
'''



'''                 mp_drawing.draw_landmarks(
                    img_rgb,  # image to draw
                    hand_landmarks,  # model output
                    mp_hands.HAND_CONNECTIONS,  # hand connections
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
'''


'''
only for one image 
        plt.figure()
        plt.imshow(img_rgb)



plt.show()
'''