
# Real-time Face Detection and Analysis

This project uses OpenCV and TensorFlow to perform real-time face detection and analysis through a webcam. It detects faces in the video stream, predicts the gender and emotion of the detected faces, and displays the results.

## Requirements

Make sure you have the following packages installed:

- OpenCV
- TensorFlow
- Keras

You can install them using pip if you haven't already:

```bash
pip install opencv-python tensorflow
```

## Models

This code assumes you have pre-trained models for gender detection and emotion detection saved as `gender_detection.keras` and `my_model2.keras`, respectively. Ensure these model files are in the same directory as your script.

## How to Run

1. Save the code into a Python file, e.g., `face_detection.py`.
2. Ensure your webcam is connected.
3. Place the model files `gender_detection.keras` and `my_model2.keras` in the same directory.
4. Run the script:

```bash
python face_detection.py
```

5. A window will appear showing the real-time video feed from your webcam. Detected faces will be marked with rectangles, and the predicted gender and emotion will be displayed below each face.
6. Press 'q' to quit the application.

## Code Explanation

1. **Imports and Initializations**:
    - Import necessary libraries (`cv2`, `tensorflow`, `keras`).
    - Initialize the webcam (`cv2.VideoCapture(0)`).
    - Load the pre-trained face detection model using OpenCV's Haar cascade classifier.

2. **Placeholder Functions for Predictions**:
    - `predict_gender(img)`: Resizes the input image, preprocesses it, loads the gender detection model, and predicts the gender.
    - `predict_emotion(img)`: Resizes the input image, preprocesses it, loads the emotion detection model, and predicts the emotion.

3. **Main Loop**:
    - Capture frames from the webcam.
    - Convert each frame to grayscale for face detection.
    - Detect faces in the grayscale frame.
    - For each detected face, extract the face region, predict gender and emotion, and display the results on the frame.
    - Show the processed frame in a window.
    - Break the loop and close the window if 'q' is pressed.

## Notes

- The code uses hardcoded class names for gender and emotion. Ensure these match the classes used during model training.
- Adjust the input image size (48x48) in the `predict_gender` and `predict_emotion` functions according to the requirements of your models.
- The emotion detection model should have the following classes: 'angry', 'happy', 'sad'. Modify the class names if your model differs.

## Troubleshooting

- If the video feed doesn't show or is very slow, ensure your webcam is properly connected and check your system's performance.
- If the models are not found, verify the file paths and names of the model files.
- Ensure the image preprocessing steps (resizing, normalization) match those used during model training.

---

This should help you get started with running and understanding the provided code.
