""" Landmark detector for processing hand landmarks. """

from src.config import config
from src.inference import LandmarkPoint
from typing import List
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2

class LandmarkDetector:
    """ 
    A class to detect hand landmarks using MediaPipe.

    Attributes:
        mp_hands (mp.solutions.hands): The MediaPipe Hands solution.
    """
    def __init__(self):
        base_options = python.BaseOptions(model_asset_path=config.mediapipe.model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options, 
            num_hands=config.mediapipe.max_num_hands,
            min_hand_detection_confidence=config.mediapipe.min_detection_confidence
        )
        self.mp_hands = vision.HandLandmarker.create_from_options(options)
        self.keypoint_connections = {
            (0, 1), (1, 2), (2, 3), (3, 4),      
            (0, 5), (5, 6), (6, 7), (7, 8),        
            (9, 10), (10, 11), (11, 12),
            (13, 14), (14, 15), (15, 16),        
            (0, 17), (17, 18), (18, 19), (19, 20),
            (5,9), (9,13), (13,17)  # Conexiones entre las bases de los dedos
        }

    def detect_landmarks(self, image)->List[LandmarkPoint]:
        """ 
        Detect hand landmarks in the given image.

        Args:
            image (numpy.ndarray): The input image in which to detect hand landmarks.
        Returns:
            List[LandmarkPoint]: A list of detected hand landmarks as LandmarkPoint instances.
        """
        if image is None or image.size == 0:
            return []
        
        # Convert the image to the format expected by MediaPipe
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

        # Perform hand landmark detection
        results = self.mp_hands.detect(image)
        landmarks = []
        if results.hand_landmarks:
            for hand_landmark in results.hand_landmarks:
                for landmark in hand_landmark:
                    landmarks.append(LandmarkPoint(x=landmark.x, y=landmark.y, z=landmark.z))
        return landmarks

    def annotate_image(self, image, landmarks: List[LandmarkPoint], point_size=5, line_thickness=2):
        """ 
        Annotate the input image with detected hand landmarks.

        Args:
            image (numpy.ndarray): The input image to annotate.
            landmarks (List[LandmarkPoint]): A list of detected hand landmarks to annotate on the image.
        Returns:
            numpy.ndarray: The annotated image with hand landmarks drawn.
        """
        annotated_image = image.copy()
        for connection in self.keypoint_connections:
            start_idx, end_idx = connection
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                start_point = (int(landmarks[start_idx].x * image.shape[1]), int(landmarks[start_idx].y * image.shape[0]))
                end_point = (int(landmarks[end_idx].x * image.shape[1]), int(landmarks[end_idx].y * image.shape[0]))
                cv2.line(annotated_image, start_point, end_point, (0, 255, 0), line_thickness)
        for landmark in landmarks:
            center = (int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0]))
            cv2.circle(annotated_image, center, point_size, (0, 0, 255), -1)
        return annotated_image

    def close(self):
        """ 
        Close the MediaPipe Hands solution to release resources.
        """
        self.mp_hands.close()
