import cv2

# Load pre-trained Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load pre-trained gender classification model
gender_net = cv2.dnn.readNetFromCaffe('deploy_gender.prototxt', 'gender_net.caffemodel')

# List of gender labels
gender_list = ['Male', 'Female']

# Function to detect faces and classify gender
def detect_and_estimate_gender(frame):
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces using Haar cascade
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Loop over detected faces
    for (x, y, w, h) in faces:
        # Extract face ROI
        face_roi = frame[y:y+h, x:x+w]

        # Preprocess face ROI for gender classification
        blob = cv2.dnn.blobFromImage(face_roi, scalefactor=1.0, size=(227, 227), mean=(78.4263377603, 87.7689143744, 114.895847746), swapRB=False)

        # Pass face ROI through gender classification model
        gender_net.setInput(blob)
        gender_preds = gender_net.forward()

        # Get predicted gender label
        gender_label = gender_list[gender_preds[0].argmax()]

        # Draw bounding box around face and label with predicted gender
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, gender_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame

# Main function to process video stream
def process_video_stream():
    # Open the video stream from webcam
    video_stream = cv2.VideoCapture(0)
    
    while True:
        # Read a frame from the video stream
        ret, frame = video_stream.read()
        if not ret:
            break
        
        # Detect and classify gender of faces in the frame
        frame_with_gender = detect_and_estimate_gender(frame.copy())
        
        # Display the frame with gender labels
        cv2.imshow('Frame with Gender', frame_with_gender)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video stream
    video_stream.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Process the video stream
    process_video_stream()
