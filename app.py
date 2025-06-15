from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import mediapipe as mp
import base64
import time
import logging
from datetime import datetime
import json
from typing import Tuple, Dict, Any
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Initialize MediaPipe solutions
mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection

# Thresholds for cheating detection
HEAD_POSE_THRESHOLD = 20
CHEATING_DURATION_THRESHOLD = 3

class CheatingDetector:
    def __init__(self):
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=2,
            refine_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.face_detector = mp_face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.7
        )
        self.cheating_state = {
            "no_face_start": None
        }

    def calculate_head_pose(self, landmarks, frame_shape: Tuple[int, int]) -> Dict[str, float]:
        model_points = np.array([
            (0.0, 0.0, 0.0),
            (0.0, -330.0, -65.0),
            (-225.0, 170.0, -135.0),
            (225.0, 170.0, -135.0),
            (-150.0, -150.0, -125.0),
            (150.0, -150.0, -125.0)
        ])

        image_points = np.array([
            [landmarks[4].x * frame_shape[1], landmarks[4].y * frame_shape[0]],
            [landmarks[152].x * frame_shape[1], landmarks[152].y * frame_shape[0]],
            [landmarks[263].x * frame_shape[1], landmarks[263].y * frame_shape[0]],
            [landmarks[33].x * frame_shape[1], landmarks[33].y * frame_shape[0]],
            [landmarks[287].x * frame_shape[1], landmarks[287].y * frame_shape[0]],
            [landmarks[57].x * frame_shape[1], landmarks[57].y * frame_shape[0]]
        ], dtype="double")

        focal_length = frame_shape[1]
        center = (frame_shape[1]/2, frame_shape[0]/2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype="double")

        dist_coeffs = np.zeros((4,1))
        (_, rotation_vector, _) = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

        rmat, _ = cv2.Rodrigues(rotation_vector)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

        return {
            "pitch": angles[0],
            "yaw": angles[1],
            "roll": angles[2]
        }

    def detect_gaze(self, landmarks, frame_shape: Tuple[int, int]) -> str:
        def get_ratio(iris, corner1, corner2):
            distance = corner2 - corner1
            if distance == 0:
                return 0.5
            return (iris - corner1) / distance

        left_iris = landmarks[468]
        left_inner = landmarks[33]
        left_outer = landmarks[133]
        left_top = landmarks[159]
        left_bottom = landmarks[145]

        right_iris = landmarks[473]
        right_inner = landmarks[362]
        right_outer = landmarks[263]
        right_top = landmarks[386]
        right_bottom = landmarks[374]

        left_x_ratio = get_ratio(left_iris.x, left_inner.x, left_outer.x)
        right_x_ratio = get_ratio(right_iris.x, right_inner.x, right_outer.x)
        x_avg = (left_x_ratio + right_x_ratio) / 2

        left_y_ratio = get_ratio(left_iris.y, left_top.y, left_bottom.y)
        right_y_ratio = get_ratio(right_iris.y, right_top.y, right_bottom.y)
        y_avg = (left_y_ratio + right_y_ratio) / 2

        head_pose = self.calculate_head_pose(landmarks, frame_shape)

        if x_avg < 0.45:
            return "left"
        elif x_avg > 0.55:
            return "right"
        elif y_avg < 0.38 and head_pose["pitch"] > 5:
            return "up"
        elif y_avg > 0.6:
            return "down"
        else:
            return "center"

    def analyze_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_shape = frame.shape[:2]

        results = {
            "face_detected": False,
            "multiple_faces": False,
            "gaze_direction": "center",
            "head_pose": {"pitch": 0, "yaw": 0, "roll": 0},
            "face_confidence": 0
        }

        face_detections = self.face_detector.process(rgb_frame)
        if face_detections.detections:
            results["face_detected"] = True
            results["face_confidence"] = face_detections.detections[0].score[0]
            if len(face_detections.detections) > 1:
                results["multiple_faces"] = True

        face_mesh_results = self.face_mesh.process(rgb_frame)
        if face_mesh_results.multi_face_landmarks:
            landmarks = face_mesh_results.multi_face_landmarks[0].landmark
            results["head_pose"] = self.calculate_head_pose(landmarks, frame_shape)
            results["gaze_direction"] = self.detect_gaze(landmarks, frame_shape)

        return results

    def detect_cheating(self, analysis: Dict[str, Any]) -> Dict[str, bool]:
        current_time = time.time()
        flags = {
            "looking_away": analysis["gaze_direction"] != "center",
            "no_face_detected": not analysis["face_detected"],
            "multiple_faces": analysis.get("multiple_faces", False),
            "extreme_head_pose": False
        }

        head_pose = analysis["head_pose"]
        if (abs(head_pose["pitch"]) > HEAD_POSE_THRESHOLD or
            abs(head_pose["yaw"]) > HEAD_POSE_THRESHOLD):
            flags["extreme_head_pose"] = True

        if flags["no_face_detected"]:
            if self.cheating_state["no_face_start"] is None:
                self.cheating_state["no_face_start"] = current_time
            elif current_time - self.cheating_state["no_face_start"] > CHEATING_DURATION_THRESHOLD:
                flags["no_face_detected"] = True
        else:
            self.cheating_state["no_face_start"] = None

        return flags

    def assess_confusion_confidence(self, analysis, cheating_flags):
        pitch = abs(analysis.get("head_pose", {}).get("pitch", 0))
        yaw = abs(analysis.get("head_pose", {}).get("yaw", 0))
        gaze = analysis.get("gaze_direction", "center")
        face_confidence = analysis.get("face_confidence", 0)

        max_angle = 40.0
        pitch_norm = min(pitch / max_angle, 1.0)
        yaw_norm = min(yaw / max_angle, 1.0)

        gaze_factor = 1.0 if gaze != "center" else 0.0
        confusion_raw = (pitch_norm + yaw_norm) / 2.0 * 0.7 + gaze_factor * 0.3
        confusion_pct = int(confusion_raw * 100)

        gaze_penalty = 0.0 if gaze == "center" else 0.4
        pose_penalty = max(pitch_norm, yaw_norm) * 0.4
        confidence_raw = face_confidence * (1.0 - gaze_penalty - pose_penalty)
        confidence_raw = max(0.0, min(confidence_raw, 1.0))
        confidence_pct = int(confidence_raw * 100)

        return {
            "confusion_percentage": confusion_pct,
            "confidence_percentage": confidence_pct
        }

class InterviewReporter:
    def __init__(self):
        self.reports = {}
    
    def generate_report(self, history, reason):
        # Handle cases where history is empty due to early termination
        if not history and ("terminated" in reason.lower() or "switched" in reason.lower()):
            logger.warning(f"Generating minimal report for termination reason: {reason} with empty history.")
            report = {
                'timestamp': datetime.now().isoformat(),
                'status': 'TERMINATED',
                'reason': reason,
                'score': 0, # Automatically a very low score for terminations without data
                'incidents': ["FORCED_TERMINATION_NO_DATA"],
                'metrics': {
                    'total_frames': 0,
                    'face_detected': 0,
                    'looking_away': 0,
                    'multiple_faces': 0,
                    'no_face': 0,
                    'avg_confidence': 0,
                    'avg_confusion': 0
                },
                'compatibility_percentage': 0, # Add compatibility percentage
                'recommendation': "REJECT" # Strong rejection if terminated with no data
            }
            report_id = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}_TERMINATED"
            self.reports[report_id] = report
            return report_id, report
            
        if not history:
            # Original condition for genuinely empty history without specific termination reason
            return None, None
            
        total_frames = len(history)
        
        # Calculate metrics
        metrics = {
            'total_frames': total_frames,
            'face_detected': sum(1 for x in history if x['analysis']['face_detected']),
            'looking_away': sum(1 for x in history if x['cheating_flags']['looking_away']),
            'multiple_faces': sum(1 for x in history if x['cheating_flags']['multiple_faces']),
            'no_face': sum(1 for x in history if x['cheating_flags']['no_face_detected']),
            'extreme_head_pose': sum(1 for x in history if x['cheating_flags']['extreme_head_pose']),
            'avg_confidence': round(sum(
                x['extra_flags']['confidence_percentage'] for x in history
            ) / total_frames, 1) if total_frames > 0 else 0,
            'avg_confusion': round(sum(
                x['extra_flags']['confusion_percentage'] for x in history
            ) / total_frames, 1) if total_frames > 0 else 0
        }
        
        # Initialize compatibility percentage
        compatibility_percentage = 100.0
        incidents = []
        status = 'COMPLETED' if reason == 'completed' else 'TERMINATED'

        # Rule 1: Tab switch is an immediate and severe incident
        if "app switched" in reason.lower():
            incidents.append("TAB_SWITCH")
            compatibility_percentage = 0 # Automatic 0% compatibility
            status = 'TERMINATED_BY_SYSTEM'
            
        # Rule 2: Strict looking away based on frame count
        # Ensure we have the actual number of frames where looking_away was true
        looking_away_frames = metrics['looking_away']
        
        if total_frames < 8:
            if looking_away_frames >= 1: # Even a single look away for very short interviews
                incidents.append("EARLY_GAZE_AVERSION")
                compatibility_percentage = 0 # Consider 100% cheating
                status = 'COMPLETED_WITH_CHEATING' if reason == 'completed' else 'TERMINATED_BY_SYSTEM'
        elif total_frames >= 10:
            if looking_away_frames > 3: # More than 3 looking away frames for longer interviews
                incidents.append("FREQUENT_GAZE_AVERSION")
                # Deduct from compatibility for frequent looking away
                compatibility_percentage -= (looking_away_frames - 3) * 5 # 5% penalty per excess looking away frame
                compatibility_percentage = max(0, compatibility_percentage) # Ensure not below 0
        
        # Rule 3: Multiple faces detected
        if metrics['multiple_faces'] > 0:
            incidents.append("MULTIPLE_FACES_DETECTED")
            compatibility_percentage = 0 # This is a critical cheating flag
            status = 'COMPLETED_WITH_CHEATING' if reason == 'completed' else 'TERMINATED_BY_SYSTEM'

        # Rule 4: No face detected for a significant portion
        if metrics['no_face'] / total_frames > 0.15: # If no face for more than 15% of frames
            incidents.append("PROLONGED_NO_FACE_DETECTED")
            # Significant penalty, or even rejection based on severity. Let's make it a significant deduction.
            compatibility_percentage -= 30 # For example, 30% deduction for prolonged no face
            compatibility_percentage = max(0, compatibility_percentage)
        elif metrics['no_face'] > 0: # Minor penalty for occasional no face
             incidents.append("OCCASIONAL_NO_FACE_DETECTED")
             compatibility_percentage -= metrics['no_face'] * 2
             compatibility_percentage = max(0, compatibility_percentage)

        # Rule 5: Extreme head pose
        if metrics['extreme_head_pose'] > 0:
            incidents.append("EXTREME_HEAD_POSE_DETECTED")
            compatibility_percentage -= metrics['extreme_head_pose'] * 1 # Small penalty for each occurrence
            compatibility_percentage = max(0, compatibility_percentage)

        # Incorporate average confidence directly into compatibility
        # Higher avg_confidence implies better visibility/engagement
        # Lower avg_confusion implies less distractions
        
        # Baseline from average confidence and confusion
        # Let's say avg_confidence directly adds to compatibility (scaled)
        # And avg_confusion deducts from compatibility (scaled)
        
        # Weighting factors (you can adjust these)
        CONFIDENCE_WEIGHT = 0.4
        CONFUSION_WEIGHT = 0.3
        
        # Initial score based on general face analysis
        base_compatibility = (metrics['avg_confidence'] * CONFIDENCE_WEIGHT) + \
                             ((100 - metrics['avg_confusion']) * CONFUSION_WEIGHT)
        
        # Combine with the initial 100% and apply deductions
        # This approach ensures that critical cheating flags can still bring it to 0
        if compatibility_percentage > 0: # Only if not already zeroed out by severe cheating
            compatibility_percentage = (compatibility_percentage / 100.0) * base_compatibility
            compatibility_percentage = min(100.0, max(0.0, compatibility_percentage)) # Ensure within 0-100

        # Round the final percentage
        compatibility_percentage = round(compatibility_percentage, 1)

        # Recommendation logic based purely on compatibility percentage
        if compatibility_percentage >= 90:
            recommendation = "Highly Compatible"
        elif compatibility_percentage >= 75:
            recommendation = "Compatible"
        elif compatibility_percentage >= 50:
            recommendation = "Moderately Compatible"
        else:
            recommendation = "Low Compatibility"
        
        # Override recommendation and status if severe incidents occurred
        if "TAB_SWITCH" in incidents or "MULTIPLE_FACES_DETECTED" in incidents or "EARLY_GAZE_AVERSION" in incidents:
            recommendation = "REJECT - Severe Cheating Detected"
            status = 'CHEATING_TERMINATED' if "TERMINATED" in status else 'CHEATING_DETECTED'
            compatibility_percentage = 0 # Ensure 0% for severe cheating

        # Create report
        report = {
            'timestamp': datetime.now().isoformat(),
            'status': status,
            'reason': reason,
            'incidents': incidents,
            'metrics': metrics,
            'compatibility_percentage': compatibility_percentage, # New field
            'recommendation': recommendation
        }
        
        # Store report
        report_id = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.reports[report_id] = report
        
        return report_id, report

# Initialize services
detector = CheatingDetector()
reporter = InterviewReporter()

@app.route('/')
def home():
    return jsonify({
        'status': 'Non-Verbal Analysis API is running!', 
        'version': '1.0',
        'endpoints': ['/analyze', '/interview/end', '/reports', '/health']
    })

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'mediapipe': 'loaded'})

@app.route('/analyze', methods=['POST'])
def analyze():
    start_time = time.time()
    try:
        if not request.json or 'image' not in request.json:
            return jsonify({"error": "Image data required"}), 400

        frame = process_image(request.json['image'])
        if frame is None:
            return jsonify({"error": "Invalid image data"}), 400

        analysis = detector.analyze_frame(frame)
        cheating_flags = detector.detect_cheating(analysis)
        extra_flags = detector.assess_confusion_confidence(analysis, cheating_flags)

        logger.info(f"Processed in {(time.time()-start_time):.2f}s - Face: {analysis['face_detected']}")

        return jsonify({
            "status": "success",
            "cheating_flags": cheating_flags,
            "analysis": analysis,
            "extra_flags": extra_flags
        })

    except Exception as e:
        logger.error(f"Analysis error: {str(e)}", exc_info=True)
        return jsonify({"error": "Processing failed"}), 500

def process_image(image_data: str) -> np.ndarray:
    try:
        if ',' in image_data:
            image_data = image_data.split(',')[1]

        img_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception as e:
        logger.error(f"Image processing error: {str(e)}")
        return None

@app.route('/interview/end', methods=['POST'])
def end_interview():
    try:
        data = request.json or {}
        analysis_history = data.get('analysis_history', [])
        reason = data.get('reason', 'unknown')
        
        # Generate report
        report_id, report = reporter.generate_report(analysis_history, reason)
        
        if not report: 
            return jsonify({"error": "No data provided"}), 400
            
        # Print to console
        print("\n=== INTERVIEW REPORT ===")
        print(json.dumps(report, indent=2))
        
        return jsonify({
            "status": "success",
            "report_id": report_id,
            "report": report
        })
        
    except Exception as e:
        logger.error(f"Interview end error: {str(e)}")
        return jsonify({"error": f"Failed to end interview: {str(e)}"}), 500

@app.route('/reports', methods=['GET'])
def get_reports():
    return jsonify({
        "status": "success",
        "reports": reporter.reports
    })

@app.route('/reports/<report_id>', methods=['GET'])
def get_report(report_id):
    report = reporter.reports.get(report_id)
    if report:
        return jsonify({
            "status": "success",
            "report": report
        })
    return jsonify({"error": "Report not found"}), 404

if __name__ == '__main__':
    # Get port from environment variable or default to 5000
    port = int(os.environ.get('PORT', 5000))
    # Set debug=False for production
    debug_mode = os.environ.get('FLASK_ENV') == 'development'
    
    app.run(host='0.0.0.0', port=port, debug=debug_mode)
