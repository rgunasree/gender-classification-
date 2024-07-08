# Gender Classification using Haarcascade and Gender Net.caffe Model

This project focuses on gender classification using Haarcascade for face detection and the Gender Net.caffe model for gender prediction. The implementation is done in Python with OpenCV and Caffe.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Setting Up Git LFS for `gender_net.caffemodel`] 
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Introduction

Gender classification is an essential task in many applications such as security systems, human-computer interaction, and demographic studies. This project utilizes Haarcascade for detecting faces in images and the Gender Net.caffe model to classify the detected faces as male or female.

## Features

- Face detection using Haarcascade.
- Gender classification using Gender Net.caffe model.
- Real-time gender classification using webcam feed.
- Easy-to-use command-line interface.

## Requirements

- Python 3.x
- OpenCV
- Caffe
- NumPy

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/gender-classification.git
    ```
2. Navigate to the project directory:
    ```sh
    cd gender-classification
    ```
3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```
4. Download the `deploy_gender.prototxt` and `gender_net.caffemodel` files and place them in the project directory.

## Setting Up Git LFS for `gender_net.caffemodel`

To properly clone this repository and handle large files like `gender_net.caffemodel`, ensure you have Git LFS installed. Follow these steps:

1. Install Git LFS:
    ```
    git lfs install
    ```

2. Clone the repository:
    ```
    git clone https://github.com/rgunasree/gender-classification-.git
    ```

3. Pull the LFS files:
    ```
    git lfs pull
    ```

Now you should have `gender_net.caffemodel` properly downloaded and ready to use.


## Usage

1. **Face Detection and Gender Classification on Images:**
    ```sh
    python classify_gender.py --image path_to_your_image.jpg
    ```
2. **Real-time Gender Classification using Webcam:**
    ```sh
    python classify_gender.py --webcam
    ```

### Command-Line Arguments

- `--image`: Path to the image file for gender classification.
- `--webcam`: Use webcam for real-time gender classification.

### Example

To classify gender from an image:
```sh
python classify_gender.py --image examples/sample.jpg
```
To use webcam for real-time gender classification:
```python classify_gender.py --webcam```

### Results

Here are some sample results attached at my repo

### Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

Fork the repository.
Create a new branch:
```
git checkout -b feature_branch
```
Make your changes and commit them:
```
git commit -m 'Add some feature'
```
Push to the branch:
```
git push origin feature_branch
```
Open a pull request.

### License

This project is licensed under the MIT License - see the LICENSE file for details.

### Acknowledgements

- The Haarcascade used for face detection.
- The Gender Net.caffe model for gender classification.
- OpenCV library for computer vision tasks.

### Code Explanation

## detect_and_estimate_gender Function
This function detects faces in a frame and estimates their gender.

- Convert frame to grayscale:
```
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
```
- Detect faces using Haar cascade:
```
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
```
- Loop over detected faces and classify gender:
```
for (x, y, w, h) in faces:
    face_roi = frame[y:y+h, x:x+w]
    blob = cv2.dnn.blobFromImage(face_roi, scalefactor=1.0, size=(227, 227), mean=(78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
    gender_net.setInput(blob)
    gender_preds = gender_net.forward()
    gender_label = gender_list[gender_preds[0].argmax()]
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(frame, gender_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
```
## process_video_stream Function
This function processes the video stream from the webcam and applies gender classification in real-time.

- Open the video stream:
```
video_stream = cv2.VideoCapture(0)
```
- Read and process each frame:
```
while True:
    ret, frame = video_stream.read()
    if not ret:
        break
    frame_with_gender = detect_and_estimate_gender(frame.copy())
    cv2.imshow('Frame with Gender', frame_with_gender)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
```
### Release resources:
```
video_stream.release()
cv2.destroyAllWindows()
```
#### Running the Code

Run the script to start the real-time gender classification:
```
python classify_gender.py --webcam
```

Replace `examples/sample.jpg` and `examples/output.jpg` with the actual paths to your images in the repository. This will embed the images directly in your `README.md` file, making it more informative and visually appealing.

```
You can copy and paste this content into your `README.md` file. Make sure to update the paths to the images and any other specific details relevant to your project.
```



