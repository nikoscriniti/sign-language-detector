README 

3 step process
''' install dependencies:
    opencv (python) --> install opencv-python==4.7.0.68
    mediapipe --> pip3 install mediapipe==0.10.13
    scikit-learn --> pip3 install  scikit-learn==1.2.0 (had to switch to 1.5.0)
'''

Three steps: 
-run collect_imgs: classifers need data (keep repeating the process with the different symbols)
-extracting the position of the image --> use a landmark detector ( to convert each image into points (around 20)) --> building this classifer 
-only classifying a landmark of a portion of the image, the input is only the position of the hand (remvoing the other stuff in the image like the pixels, person, objects in the background)


# used: Random forest is a commonly-used machine learning algorithm,

TEST:

remmber when youre going to test: run files one after another at frist: 
first: collect...
second: create...
third: train...
fourth: test_live

When collecting, remember to start with a, b, c... and so on, it will remember as such when you go to test it live based on the dictonary of values 