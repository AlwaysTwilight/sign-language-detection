# from keras.models import model_from_json
# import cv2
# import numpy as np

# json_file = open("G:\sign_language_detection\custom\customsignlanguagedetectionmodel48x48.json", "r")
# model_json = json_file.read()
# json_file.close()
# model = model_from_json(model_json)
# model.load_weights("model.h5")

# def extract_features(image):
#     feature = np.array(image)
#     feature = feature.reshape(1,48,48,1)
#     return feature/255.0

# cap = cv2.VideoCapture(0)
# label = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'blank']
# while True:
#     _,frame = cap.read()
#     cv2.rectangle(frame,(0,40),(300,300),(0, 165, 255),1)
#     cropframe=frame[40:300,0:300]
#     cropframe=cv2.cvtColor(cropframe,cv2.COLOR_BGR2GRAY)
#     cropframe = cv2.resize(cropframe,(48,48))
#     cropframe = extract_features(cropframe)
#     pred = model.predict(cropframe) 
#     prediction_label = label[pred.argmax()]
#     cv2.rectangle(frame, (0,0), (300, 40), (0, 165, 255), -1)
#     if prediction_label == 'blank':
#         cv2.putText(frame, " ", (10, 30),cv2.FONT_HERSHEY_SIMPLEX,1, (255, 255, 255),2,cv2.LINE_AA)
#     else:
#         accu = "{:.2f}".format(np.max(pred)*100)
#         cv2.putText(frame, f'{prediction_label}  {accu}%', (10, 30),cv2.FONT_HERSHEY_SIMPLEX,1, (255, 255, 255),2,cv2.LINE_AA)
#     cv2.imshow("output",frame)
#     cv2.waitKey(27)
    
#     cap.release()
#     cv2.destroyAllWindows()

from keras.models import model_from_json
import cv2
import numpy as np

# Load the model architecture and weights
json_file_path = "G:\\sign_language_detection\\custom\\customsignlanguagedetectionmodel48x48.json"
weights_file_path = "G:\sign_language_detection\custom\model.h5"

# Load model architecture
with open(json_file_path, "r") as json_file:
    model_json = json_file.read()

model = model_from_json(model_json)
model.load_weights(weights_file_path)

def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

# Open a connection to the camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video capture.")
    exit()

label = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'blank']

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)

    # Define ROI on the right side
    frame_height, frame_width = frame.shape[:2]
    roi_width = 300
    roi_height = 260  # You can adjust the height if needed
    roi_x1 = frame_width - roi_width
    roi_x2 = frame_width
    roi_y1 = 40
    roi_y2 = roi_y1 + roi_height

    # Draw the ROI rectangle and process the ROI
    cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 165, 255), 1)
    cropframe = frame[roi_y1:roi_y2, roi_x1:roi_x2]
    cropframe = cv2.cvtColor(cropframe, cv2.COLOR_BGR2GRAY)
    cropframe = cv2.resize(cropframe, (48, 48))
    cropframe = extract_features(cropframe)
    pred = model.predict(cropframe)
    prediction_label = label[pred.argmax()]

    # Display the prediction on the frame
    cv2.rectangle(frame, (roi_x1, roi_y1 - 40), (roi_x2, roi_y1), (0, 165, 255), -1)
    if prediction_label == 'blank':
        cv2.putText(frame, " ", (roi_x1 + 10, roi_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    else:
        accu = "{:.2f}".format(np.max(pred) * 100)
        cv2.putText(frame, f'{prediction_label}  {accu}%', (roi_x1 + 10, roi_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow("output", frame)
    
    if cv2.waitKey(27) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

