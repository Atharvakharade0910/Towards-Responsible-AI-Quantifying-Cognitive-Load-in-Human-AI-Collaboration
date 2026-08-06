from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
import onnxruntime as ort


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_FILENAME = (
    "facial_expression_recognition_mobilefacenet_2022july.onnx"
)

DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / MODEL_FILENAME


# The order must match the exact output order of the OpenCV model.
EMOTION_LABELS = (
    "angry",
    "disgust",
    "fearful",
    "happy",
    "neutral",
    "sad",
    "surprised",
)


# Standard five-point facial template used for 112 × 112 aligned faces.
#
# Point order:
# 1. Image-left eye
# 2. Image-right eye
# 3. Nose tip
# 4. Image-left mouth corner
# 5. Image-right mouth corner
FACE_ALIGNMENT_TEMPLATE = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


class FacialExpressionAnalyzer:
    """
    Detect, align and classify visible facial expressions.

    The component is independent of eye tracking. It receives a BGR image
    and returns one facial-expression result.
    """

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        minimum_detection_confidence: float = 0.60,
        minimum_tracking_confidence: float = 0.60,
    ) -> None:
        self.model_path = Path(model_path).resolve()

        if not self.model_path.exists():
            raise FileNotFoundError(
                "Facial-expression model was not found at:\n"
                f"{self.model_path}\n\n"
                "Run setup_facial_model.py before starting the application."
            )

        if not 0.0 <= minimum_detection_confidence <= 1.0:
            raise ValueError(
                "minimum_detection_confidence must be between 0 and 1."
            )

        if not 0.0 <= minimum_tracking_confidence <= 1.0:
            raise ValueError(
                "minimum_tracking_confidence must be between 0 and 1."
            )

        self._session = self._create_onnx_session()

        self._input = self._session.get_inputs()[0]
        self._input_name = self._input.name

        self._output_names = [
            output.name for output in self._session.get_outputs()
        ]

        self._validate_model_structure()

        # MediaPipe detects and tracks the face landmarks.
        #
        # static_image_mode=False allows MediaPipe to use tracking between
        # sequential frames instead of performing full detection every time.
        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=minimum_detection_confidence,
            min_tracking_confidence=minimum_tracking_confidence,
        )

    def _create_onnx_session(self) -> ort.InferenceSession:
        """
        Load the model using the CPU execution provider.
        """

        options = ort.SessionOptions()

        # Restricting the model to one inference thread helps prevent the
        # facial module from taking every CPU core away from eye tracking.
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1

        options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        return ort.InferenceSession(
            str(self.model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )

    def _validate_model_structure(self) -> None:
        """
        Confirm that the loaded model has a usable image input.
        """

        input_shape = self._input.shape

        if len(input_shape) != 4:
            raise RuntimeError(
                "Expected a four-dimensional ONNX input such as "
                f"[batch, channels, height, width], but received {input_shape}."
            )

        if not self._output_names:
            raise RuntimeError(
                "The ONNX model does not contain any output nodes."
            )

    @staticmethod
    def _average_landmarks(
        landmarks: Any,
        landmark_indices: tuple[int, ...],
        image_width: int,
        image_height: int,
    ) -> np.ndarray:
        """
        Calculate the average pixel position of selected landmarks.
        """

        points = []

        for index in landmark_indices:
            landmark = landmarks[index]

            points.append(
                [
                    landmark.x * image_width,
                    landmark.y * image_height,
                ]
            )

        return np.mean(
            np.asarray(points, dtype=np.float32),
            axis=0,
        )

    def _extract_alignment_points(
        self,
        landmarks: Any,
        image_width: int,
        image_height: int,
    ) -> np.ndarray:
        """
        Convert MediaPipe landmarks into five alignment points.
        """

        # These pairs represent the eye areas.
        image_left_eye = self._average_landmarks(
            landmarks,
            (33, 133),
            image_width,
            image_height,
        )

        image_right_eye = self._average_landmarks(
            landmarks,
            (362, 263),
            image_width,
            image_height,
        )

        nose_tip = self._average_landmarks(
            landmarks,
            (1,),
            image_width,
            image_height,
        )

        image_left_mouth = self._average_landmarks(
            landmarks,
            (61,),
            image_width,
            image_height,
        )

        image_right_mouth = self._average_landmarks(
            landmarks,
            (291,),
            image_width,
            image_height,
        )

        return np.asarray(
            [
                image_left_eye,
                image_right_eye,
                nose_tip,
                image_left_mouth,
                image_right_mouth,
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _points_are_valid(
        points: np.ndarray,
        image_width: int,
        image_height: int,
    ) -> bool:
        """
        Reject invalid or extreme landmark coordinates.
        """

        if points.shape != (5, 2):
            return False

        if not np.isfinite(points).all():
            return False

        # A small margin is allowed because landmarks can occasionally appear
        # slightly outside the image when the face is near the camera border.
        horizontal_margin = image_width * 0.10
        vertical_margin = image_height * 0.10

        x_valid = np.logical_and(
            points[:, 0] >= -horizontal_margin,
            points[:, 0] <= image_width + horizontal_margin,
        )

        y_valid = np.logical_and(
            points[:, 1] >= -vertical_margin,
            points[:, 1] <= image_height + vertical_margin,
        )

        return bool(np.all(x_valid) and np.all(y_valid))

    @staticmethod
    def _align_face(
        frame_bgr: np.ndarray,
        source_points: np.ndarray,
    ) -> np.ndarray | None:
        """
        Transform the detected face into the model's 112 × 112 template.
        """

        transformation, _ = cv2.estimateAffinePartial2D(
            source_points,
            FACE_ALIGNMENT_TEMPLATE,
            method=cv2.LMEDS,
        )

        if transformation is None:
            return None

        aligned_face = cv2.warpAffine(
            frame_bgr,
            transformation,
            (112, 112),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

        return aligned_face

    @staticmethod
    def _prepare_model_input(
        aligned_face_bgr: np.ndarray,
    ) -> np.ndarray:
        """
        Apply the preprocessing expected by the OpenCV FER model.

        Processing:
        BGR → RGB
        uint8 → float32
        [0, 255] → [0, 1]
        normalize using mean=0.5 and standard deviation=0.5
        HWC → CHW
        add batch dimension
        """

        if aligned_face_bgr.shape[:2] != (112, 112):
            aligned_face_bgr = cv2.resize(
                aligned_face_bgr,
                (112, 112),
                interpolation=cv2.INTER_LINEAR,
            )

        face_rgb = cv2.cvtColor(
            aligned_face_bgr,
            cv2.COLOR_BGR2RGB,
        )

        normalized = face_rgb.astype(np.float32) / 255.0

        normalized = (normalized - 0.5) / 0.5

        # Convert:
        # height × width × channels
        # to
        # channels × height × width
        input_tensor = np.transpose(
            normalized,
            (2, 0, 1),
        )

        input_tensor = np.expand_dims(
            input_tensor,
            axis=0,
        )

        return np.ascontiguousarray(
            input_tensor,
            dtype=np.float32,
        )

    @staticmethod
    def _convert_scores_to_probabilities(
        raw_scores: np.ndarray,
    ) -> np.ndarray:
        """
        Convert model output scores to normalized relative scores.

        The resulting values add up to 1. These should be treated as model
        scores, not guaranteed calibrated psychological probabilities.
        """

        scores = np.asarray(
            raw_scores,
            dtype=np.float32,
        ).reshape(-1)

        if scores.size != len(EMOTION_LABELS):
            raise RuntimeError(
                "Expected seven facial-expression scores, "
                f"but the model returned {scores.size}."
            )

        # If the output already looks like probabilities, normalize it safely.
        if np.all(scores >= 0.0) and np.isclose(
            np.sum(scores),
            1.0,
            atol=1e-3,
        ):
            total = float(np.sum(scores))

            if total <= 0.0:
                return np.full(
                    len(EMOTION_LABELS),
                    1.0 / len(EMOTION_LABELS),
                    dtype=np.float32,
                )

            return scores / total

        # Otherwise treat the values as logits and apply a stable softmax.
        shifted_scores = scores - np.max(scores)
        exponentials = np.exp(shifted_scores)
        denominator = float(np.sum(exponentials))

        if denominator <= 0.0 or not np.isfinite(denominator):
            raise RuntimeError(
                "The model produced invalid expression scores."
            )

        return exponentials / denominator

    def analyze_frame(
        self,
        frame_bgr: np.ndarray,
    ) -> dict[str, Any]:
        """
        Analyze one BGR webcam frame.

        Returns:
            {
                "face_detected": bool,
                "emotion": str,
                "model_confidence": float,
                "scores": dict,
                "inference_ms": float,
                "processing_ms": float
            }
        """

        processing_started = time.perf_counter()

        if frame_bgr is None:
            return self._invalid_result(
                reason="frame_is_none",
                processing_started=processing_started,
            )

        if not isinstance(frame_bgr, np.ndarray):
            return self._invalid_result(
                reason="frame_is_not_numpy_array",
                processing_started=processing_started,
            )

        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            return self._invalid_result(
                reason="invalid_frame_shape",
                processing_started=processing_started,
            )

        if frame_bgr.size == 0:
            return self._invalid_result(
                reason="empty_frame",
                processing_started=processing_started,
            )

        image_height, image_width = frame_bgr.shape[:2]

        frame_rgb = cv2.cvtColor(
            frame_bgr,
            cv2.COLOR_BGR2RGB,
        )

        detection_result = self._face_mesh.process(frame_rgb)

        if not detection_result.multi_face_landmarks:
            return self._invalid_result(
                reason="no_face",
                processing_started=processing_started,
            )

        face_landmarks = (
            detection_result.multi_face_landmarks[0].landmark
        )

        alignment_points = self._extract_alignment_points(
            face_landmarks,
            image_width,
            image_height,
        )

        if not self._points_are_valid(
            alignment_points,
            image_width,
            image_height,
        ):
            return self._invalid_result(
                reason="invalid_landmarks",
                processing_started=processing_started,
            )

        aligned_face = self._align_face(
            frame_bgr,
            alignment_points,
        )

        if aligned_face is None:
            return self._invalid_result(
                reason="alignment_failed",
                processing_started=processing_started,
            )

        input_tensor = self._prepare_model_input(
            aligned_face,
        )

        inference_started = time.perf_counter()

        outputs = self._session.run(
            self._output_names,
            {
                self._input_name: input_tensor,
            },
        )

        inference_ms = (
            time.perf_counter() - inference_started
        ) * 1000.0

        if not outputs:
            raise RuntimeError(
                "The ONNX model returned no output."
            )

        probabilities = self._convert_scores_to_probabilities(
            outputs[0],
        )

        dominant_index = int(np.argmax(probabilities))

        expression_scores = {
            emotion: round(float(probabilities[index]), 6)
            for index, emotion in enumerate(EMOTION_LABELS)
        }

        processing_ms = (
            time.perf_counter() - processing_started
        ) * 1000.0

        return {
            "face_detected": True,
            "emotion": EMOTION_LABELS[dominant_index],
            "model_confidence": round(
                float(probabilities[dominant_index]),
                6,
            ),
            "scores": expression_scores,
            "inference_ms": round(inference_ms, 3),
            "processing_ms": round(processing_ms, 3),
            "reason": "success",
        }

    @staticmethod
    def _invalid_result(
        reason: str,
        processing_started: float,
    ) -> dict[str, Any]:
        """
        Produce a consistent result when a valid face is unavailable.
        """

        processing_ms = (
            time.perf_counter() - processing_started
        ) * 1000.0

        return {
            "face_detected": False,
            "emotion": "unknown",
            "model_confidence": 0.0,
            "scores": {
                emotion: 0.0
                for emotion in EMOTION_LABELS
            },
            "inference_ms": 0.0,
            "processing_ms": round(processing_ms, 3),
            "reason": reason,
        }

    def close(self) -> None:
        """
        Release MediaPipe resources.
        """

        if self._face_mesh is not None:
            self._face_mesh.close()

    def __enter__(self) -> "FacialExpressionAnalyzer":
        return self

    def __exit__(
        self,
        exception_type: Any,
        exception_value: Any,
        traceback: Any,
    ) -> None:
        self.close()


def test_image(
    image_path: str | Path,
    model_path: str | Path,
) -> int:
    """
    Run one local image through the complete facial-expression pipeline.
    """

    image_path = Path(image_path).resolve()

    if not image_path.exists():
        print(f"Test image was not found: {image_path}")
        return 1

    frame = cv2.imread(str(image_path))

    if frame is None:
        print(
            "OpenCV could not read the image. "
            "Use a JPG, JPEG or PNG file."
        )
        return 1

    try:
        with FacialExpressionAnalyzer(
            model_path=model_path,
        ) as analyzer:
            result = analyzer.analyze_frame(frame)

    except Exception as error:
        print("\nFACIAL-EXPRESSION TEST FAILED")
        print(f"Error: {error}")
        return 1

    print("\nFacial-expression result:")
    print(
        json.dumps(
            result,
            indent=2,
        )
    )

    if not result["face_detected"]:
        print(
            "\nNo usable face was detected. "
            "Try a clear, front-facing and well-lit image."
        )
        return 1

    print("\nSTEP 2 COMPLETED SUCCESSFULLY")
    print(
        "The face was detected, aligned and classified "
        "using the ONNX model."
    )

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Test the CogniTrack facial-expression analyzer "
            "using one image."
        )
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Path to a clear test image containing one face.",
    )

    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL_PATH),
        help="Path to the facial-expression ONNX model.",
    )

    arguments = parser.parse_args()

    return test_image(
        image_path=arguments.image,
        model_path=arguments.model,
    )


if __name__ == "__main__":
    raise SystemExit(main())