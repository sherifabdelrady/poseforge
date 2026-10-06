"""PoseForge — Real-Time Pose Estimation & Gesture Recognition"""
import cv2, numpy as np, mediapipe as mp, argparse

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

GESTURES = {
    "T_POSE": lambda lm: abs(lm[mp_pose.PoseLandmark.LEFT_WRIST].y - lm[mp_pose.PoseLandmark.LEFT_SHOULDER].y) < 0.05 and abs(lm[mp_pose.PoseLandmark.RIGHT_WRIST].y - lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].y) < 0.05,
    "HANDS_UP": lambda lm: lm[mp_pose.PoseLandmark.LEFT_WRIST].y < lm[mp_pose.PoseLandmark.NOSE].y and lm[mp_pose.PoseLandmark.RIGHT_WRIST].y < lm[mp_pose.PoseLandmark.NOSE].y,
}

def angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a-b, c-b
    cos = np.dot(ba,bc) / (np.linalg.norm(ba)*np.linalg.norm(bc)+1e-8)
    return float(np.degrees(np.arccos(np.clip(cos,-1,1))))

def classify_gesture(landmarks):
    lm = landmarks.landmark
    for name, fn in GESTURES.items():
        try:
            if fn(lm): return name
        except: pass
    return "UNKNOWN"

def run(source=0):
    cap = cv2.VideoCapture(source)
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)
            if results.pose_landmarks:
                mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                lm = results.pose_landmarks.landmark
                elbow_angle = angle([lm[mp_pose.PoseLandmark.LEFT_SHOULDER].x, lm[mp_pose.PoseLandmark.LEFT_SHOULDER].y],
                                    [lm[mp_pose.PoseLandmark.LEFT_ELBOW].x,    lm[mp_pose.PoseLandmark.LEFT_ELBOW].y],
                                    [lm[mp_pose.PoseLandmark.LEFT_WRIST].x,    lm[mp_pose.PoseLandmark.LEFT_WRIST].y])
                gesture = classify_gesture(results.pose_landmarks)
                cv2.putText(frame, f"Elbow: {elbow_angle:.0f} deg", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                cv2.putText(frame, f"Gesture: {gesture}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)
            cv2.imshow("PoseForge", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"): break
    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--source", default="0")
    a = p.parse_args(); run(int(a.source) if a.source.isdigit() else a.source)
