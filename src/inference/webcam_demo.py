"""Real-time inference demo using webcam."""

import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path

from src.models.random_forest import RandomForestClassifier
from src.inference import LandmarkProcessor
from src.data import DataLoader


class WebcamDemo:
    """Real-time ASL classification demo."""

    def __init__(self, model_path: Path, min_confidence: float = 0.4) -> None:
        """Initializes the demo.

        Args:
            model_path: Path to the .pkl model.
            min_confidence: Minimum confidence to display prediction.
        """
        self.model = RandomForestClassifier.load(model_path)
        self.processor = LandmarkProcessor()
        self.min_confidence = min_confidence

        # Obtener nombres de clases
        loader = DataLoader()
        loader.load_data()
        self.class_names = loader.get_class_names()

        # Inicializar MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        )

    def run(self, camera_id: int = 0) -> None:
        """Runs the webcam demo."""
        cap = cv2.VideoCapture(camera_id)

        if not cap.isOpened():
            print("Error: Could not open camera.")
            return

        print("Demo ASL started. Press 'q' to exit.")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(rgb_frame)

                prediction = None
                confidence = 0.0

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        self.mp_drawing.draw_landmarks(
                            frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                        )

                        normalized = self.processor.process_mediapipe_landmarks(hand_landmarks)
                        if normalized is not None:
                            proba = self.model.predict_proba(normalized.reshape(1, -1))
                            class_idx = np.argmax(proba[0])
                            confidence = proba[0, class_idx]
                            prediction = self.class_names[class_idx]

                self._draw_prediction(frame, prediction, confidence)
                cv2.imshow("SignCapture - ASL Demo", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.hands.close()

    def _draw_prediction(self, frame: np.ndarray, prediction: str | None, confidence: float) -> None:
        """Draws the prediction on the frame."""
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (250, 90), (0, 0, 0), -1)
        frame[:] = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)

        if prediction is not None and confidence >= self.min_confidence:
            cv2.putText(frame, prediction, (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 3)
            cv2.putText(frame, f"{confidence:.0%}", (150, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        else:
            cv2.putText(frame, "?", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (128, 128, 128), 3)

        cv2.putText(frame, "Press 'q' to exit", (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
