# An auxiliary array in Python is an additional array (or list) used to assist in algorithms or data manipulations. It's often used to temporarily store intermediate results, sorted data, or for duplication during operations like sorting, merging, or partitioning.


# problem: ran into issue of only noticing "A"
    # fix --> data was not diverse enough, need more data to predict more letters
# problem: ran into warning: " UserWarning: SymbolDatabase.GetPrototype() is deprecated. Please use message_factory.GetMessageClass() instead. SymbolDatabase.GetPrototype() will be removed soon."
    # fixed: added: warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf')
# problem: text wasnt staying on screen
    # fixed: moved text code (line) outside the hand block
import pickle

import cv2
import mediapipe as mp
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf')

model_dict = pickle.load(open('/Users/crinitinikos/Desktop/summer coding porjects/sign language detector/data/model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0) # have to do video capture 0 or 2
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3) # true is looking for a still hand movment, false is looking for a moving hand movment (working with a more video streming platform)

labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 
    19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z',
    26: '1', 27: '2', 28: '3', 29: '4', 30: '5', 31: '6', 32: '7', 33: '8', 34: '9'
}
typed_text = ""
last_saved_character = None  # To store the last saved character
gesture_added = False  # To track whether the current gesture has been added


while True:

    data_array = []
    x_ = []
    y_ = []

    ret, frame = cap.read()

    H, W, _ = frame.shape # converting from float to integer

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_array.append(x - min(x_))
                data_array.append(y - min(y_))

        x1 = int(min(x_) * W) - 10 # make it an integer
        y1 = int(min(y_) * H) - 10

        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        prediction = model.predict([np.asarray(data_array)])  # Predict the letter
        predicted_character = labels_dict[int(prediction[0])]  # Get the predicted character

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4) # set color to blach 0 0 0. 4 is the thickness value of the winodw
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,cv2.LINE_AA)

        # Append only if it's not added yet for the current gesture
        if predicted_character != last_saved_character:
            typed_text += predicted_character  # Append the letter to the typed text
            last_saved_character = predicted_character  # Set the last saved character
            gesture_added = True  # Mark the gesture as added
        elif not gesture_added:
            # Only add once if the gesture hasn't been added yet
            gesture_added = True

        # Display the typed text on the frame
        
    else:
        # If no hand is detected, reset the gesture-added flag to allow new gesture capture
        gesture_added = False
        last_saved_character = None  # Reset last saved character

    cv2.putText(frame, f'Typed Text: {typed_text}', (25, 600), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)

    # Show the video frame
    cv2.imshow('Hand Gesture Recognition', frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



    

cap.release()
cv2.destroyAllWindows()

with open("/Users/crinitinikos/Desktop/summer coding porjects/sign language detector/typed_text.txt", "a") as f:
    f.write(typed_text + "\n")  # Append text to a new line
    print("Typed text saved to '/Users/crinitinikos/Desktop/summer coding porjects/sign language detector/typed_text.txt'")