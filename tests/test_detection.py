
import unittest
import cv2
import numpy as np
from src.detection import create_detector

class TestDetection(unittest.TestCase):
    def test_detector_creation(self):
        detector = create_detector("yolo")
        self.assertIsNotNone(detector)

    def test_empty_detection(self):
        detector = create_detector("yolo")
        if not detector.is_initialized():
            self.skipTest("Detector no inicializado")

        # Crear imagen vacía para test
        empty_img = np.zeros((100, 100, 3), dtype=np.uint8)
        detections = detector.detect(empty_img)

        # No debería detectar nada en una imagen vacía
        self.assertEqual(len(detections), 0)

if __name__ == '__main__':
    unittest.main()
