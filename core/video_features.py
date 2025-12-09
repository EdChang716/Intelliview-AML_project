# core/video_features.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Union

import cv2
import numpy as np
import mediapipe as mp


def _euclidean(p1, p2):
    return float(np.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2))


def extract_video_features(video_path: Union[str, Path], max_frames: int = 600) -> Dict[str, Any]:
    """
    從一個 .webm / .mp4 等影片檔抽出「面試行為」相關的視覺特徵與分數。

    回傳欄位（全部在同一層 dict）：

    基本資訊：
    - duration_sec        : 影片長度（秒）
    - frame_count         : 總 frame 數
    - fps                 : 每秒幀數
    - face_presence_ratio : 有偵測到臉的 frame 比例（0–1）

    高階分數（0–10，愈高愈好）：
    - eye_contact_score           : 眼神接觸品質（中心 + 穩定）
    - facial_positivity_score     : 表情親和度（微笑比例）
    - posture_stability_score     : 上半身穩定度
    - gesture_expressiveness_score: 手勢表達自然度
    - head_nodding_score          : 點頭 / 強調的自然度
    - fidgeting_score             : 小動作少 → 高分

    一些 raw 比例（0–1，方便之後 debug / 畫圖）：
    - eye_contact_ratio   : 被判定為「看著鏡頭」的 frame 比例
    - smile_ratio         : 被判定為「有微笑」的 frame 比例
    """

    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {
            "duration_sec": 0.0,
            "frame_count": 0,
            "fps": 0.0,
            "face_presence_ratio": 0.0,
            "eye_contact_ratio": 0.0,
            "smile_ratio": 0.0,
            "eye_contact_score": 0.0,
            "facial_positivity_score": 0.0,
            "posture_stability_score": 0.0,
            "gesture_expressiveness_score": 0.0,
            "head_nodding_score": 0.0,
            "fidgeting_score": 0.0,
        }

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_sec = frame_count / fps if fps > 0 else 0.0

    # 控制最多讀多少 frame（避免超長影片）
    if frame_count > 0:
        frames_to_read = min(frame_count, max_frames)
        step = max(frame_count // frames_to_read, 1)
    else:
        frames_to_read = 0
        step = 1

    mp_face_mesh = mp.solutions.face_mesh
    mp_pose = mp.solutions.pose

    face_presence_frames = 0
    sampled_frames = 0

    # --- for eye contact / 表情 ---
    face_centers = []          # (cx, cy) in [0, 1]
    eye_contact_frames = 0
    smile_flags = []

    # --- for posture / gestures / fidgeting ---
    torso_centers = []         # (x, y) of mid-shoulder
    gesture_values = []        # 手與軀幹距離
    head_y_positions = []      # 頭（鼻子）的 y
    body_move_deltas = []      # 軀幹移動量（fidgeting 用）

    prev_torso_center = None

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        static_image_mode=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh, mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:

        i = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if i % step != 0:
                i += 1
                continue
            i += 1
            sampled_frames += 1

            h, w = frame.shape[:2]
            if h == 0 or w == 0:
                continue

            # OpenCV BGR -> MediaPipe 需要 RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            face_res = face_mesh.process(rgb)
            pose_res = pose.process(rgb)

            # =======================
            # 1) Face 相關（eye contact / smile）
            # =======================
            face_landmarks = None
            if face_res.multi_face_landmarks:
                face_landmarks = face_res.multi_face_landmarks[0]
                face_presence_frames += 1

                xs = [lm.x for lm in face_landmarks.landmark]
                ys = [lm.y for lm in face_landmarks.landmark]
                min_x, max_x = min(xs), max(xs)
                min_y, max_y = min(ys), max(ys)
                cx = (min_x + max_x) / 2.0
                cy = (min_y + max_y) / 2.0
                face_centers.append((cx, cy))

                # ---- Eye contact：臉大致在畫面中央就當作有 eye contact ----
                # （很粗略，但對 webcam 面試 scenario 其實蠻實用）
                if abs(cx - 0.5) < 0.15 and abs(cy - 0.5) < 0.2:
                    eye_contact_frames += 1

                # ---- Smile / facial positivity （用嘴角寬高比作 proxy）----
                # 這裡用常見的 mediapipe mouth landmark index：
                # 左嘴角 61, 右嘴角 291, 上唇 0, 下唇 17（可再微調）
                lm = face_landmarks.landmark
                try:
                    left_mouth = lm[61]
                    right_mouth = lm[291]
                    top_mouth = lm[0]
                    bottom_mouth = lm[17]

                    mouth_w = _euclidean(left_mouth, right_mouth)
                    mouth_h = _euclidean(top_mouth, bottom_mouth)

                    smile_ratio = mouth_w / (mouth_h + 1e-6)
                    # ratio 大代表嘴角往兩邊拉開，比較像在笑
                    # 這邊用一個粗略閾值判斷是否「在微笑」
                    is_smiling = smile_ratio > 1.8
                    smile_flags.append(1 if is_smiling else 0)
                except Exception:
                    # landmark index 出問題或偵測異常時就略過
                    smile_flags.append(0)
            else:
                smile_flags.append(0)

            # =======================
            # 2) Pose 相關（posture / gesture / fidgeting / nodding）
            # =======================
            if pose_res.pose_landmarks:
                plm = pose_res.pose_landmarks.landmark
                # 肩膀與手腕 index：
                # https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
                try:
                    left_shoulder = plm[11]
                    right_shoulder = plm[12]
                    left_wrist = plm[15]
                    right_wrist = plm[16]
                    nose = plm[0]
                except IndexError:
                    left_shoulder = right_shoulder = left_wrist = right_wrist = nose = None

                # ---- Torso center ----
                if left_shoulder and right_shoulder:
                    torso_x = (left_shoulder.x + right_shoulder.x) / 2.0
                    torso_y = (left_shoulder.y + right_shoulder.y) / 2.0
                    torso_center = (torso_x, torso_y)
                    torso_centers.append(torso_center)

                    # fidgeting：看 torso 移動量
                    if prev_torso_center is not None:
                        dx = torso_center[0] - prev_torso_center[0]
                        dy = torso_center[1] - prev_torso_center[1]
                        body_move_deltas.append(float(np.sqrt(dx * dx + dy * dy)))
                    prev_torso_center = torso_center

                    # gesture expressiveness：手與軀幹距離
                    dists = []
                    if left_wrist:
                        dists.append(
                            float(
                                np.sqrt(
                                    (left_wrist.x - torso_x) ** 2
                                    + (left_wrist.y - torso_y) ** 2
                                )
                            )
                        )
                    if right_wrist:
                        dists.append(
                            float(
                                np.sqrt(
                                    (right_wrist.x - torso_x) ** 2
                                    + (right_wrist.y - torso_y) ** 2
                                )
                            )
                        )
                    if dists:
                        gesture_values.append(float(np.mean(dists)))

                # head nodding：用鼻子 y 軸位置粗略看「上下點頭」
                if nose:
                    head_y_positions.append(nose.y)

    cap.release()

    # ========== 統計與分數計算 ==========

    if sampled_frames == 0:
        return {
            "duration_sec": duration_sec,
            "frame_count": frame_count,
            "fps": fps,
            "face_presence_ratio": 0.0,
            "eye_contact_ratio": 0.0,
            "smile_ratio": 0.0,
            "eye_contact_score": 0.0,
            "facial_positivity_score": 0.0,
            "posture_stability_score": 0.0,
            "gesture_expressiveness_score": 0.0,
            "head_nodding_score": 0.0,
            "fidgeting_score": 0.0,
        }

    # 1) 臉出現比例
    face_presence_ratio = face_presence_frames / sampled_frames

    # 2) eye contact ratio
    eye_contact_ratio = eye_contact_frames / max(1, face_presence_frames)

    # 3) smile ratio
    smile_ratio = sum(smile_flags) / max(1, len(smile_flags))

    # 4) posture stability：torso center 位置的 std 越小越穩
    if len(torso_centers) >= 2:
        torso_arr = np.array(torso_centers)  # shape [N, 2]
        std_xy = np.std(torso_arr, axis=0)
        # 約略來說，0.02 左右的位移就算蠻明顯
        torso_disp = float(np.linalg.norm(std_xy))
        torso_norm = min(torso_disp / 0.08, 1.0)  # 0 ~ 1
        posture_stability_score = (1.0 - torso_norm) * 10.0
    else:
        posture_stability_score = 0.0

    # 5) gesture expressiveness：手與軀幹距離的平均
    if gesture_values:
        gesture_mean = float(np.mean(gesture_values))
        # 希望有「適度」的手勢，太少 / 太多都扣分
        # 0.02 以下幾乎不動，0.05~0.2 覺得剛好，>0.25 太多
        if gesture_mean <= 0.02:
            raw = gesture_mean / 0.02  # 0~1
        elif gesture_mean <= 0.18:
            raw = 1.0
        else:
            raw = max(0.0, 1.0 - (gesture_mean - 0.18) / 0.18)
        gesture_expressiveness_score = raw * 10.0
    else:
        gesture_expressiveness_score = 0.0

    # 6) head nodding：用 head_y_positions 的一階差分抓「下去又上來」的 pattern
    nod_count = 0
    if len(head_y_positions) >= 3:
        dy = np.diff(np.array(head_y_positions))
        state = "neutral"
        thresh_down = 0.006  # 往下超過門檻
        thresh_up = 0.006    # 往上超過門檻

        for d in dy:
            if state == "neutral":
                if d < -thresh_down:
                    state = "down"
            elif state == "down":
                if d > thresh_up:
                    nod_count += 1
                    state = "neutral"

    # 轉成「每 30 秒幾次點頭」，再 map 到 0–10（2~4 次 / 30 秒最理想）
    if duration_sec > 0:
        nods_per_30s = nod_count / (duration_sec / 30.0)
    else:
        nods_per_30s = 0.0

    ideal_nods = 3.0  # 理想大概 2~4 次
    diff = abs(nods_per_30s - ideal_nods)
    head_nodding_score = max(0.0, 1.0 - diff / 4.0) * 10.0

    # 7) fidgeting：torso 移動量（body_move_deltas 越大 -> 越緊張）
    if body_move_deltas:
        move_mean = float(np.mean(body_move_deltas))
        # 正常說話時，torso 每 frame 位移大概在 0.001 ~ 0.01 之間
        move_norm = min(move_mean / 0.02, 1.0)
        fidgeting_score = (1.0 - move_norm) * 10.0
    else:
        fidgeting_score = 0.0

    # 8) eye contact score：以 eye_contact_ratio + center 穩定度為 proxy
    if face_centers:
        centers_arr = np.array(face_centers)
        std_center = float(np.linalg.norm(np.std(centers_arr, axis=0)))
        # std 小代表比較穩；0.02 左右算穩，>0.08 算很晃
        center_norm = min(std_center / 0.08, 1.0)
        # eye contact ratio + 中心穩定度一起考慮
        eye_contact_score = (
            0.7 * eye_contact_ratio + 0.3 * (1.0 - center_norm)
        ) * 10.0
    else:
        eye_contact_score = 0.0

    # 9) facial positivity score：純粹用 smile_ratio
    facial_positivity_score = min(smile_ratio / 0.3, 1.0) * 10.0  # 30% 時間在笑就算滿分

    return {
        "duration_sec": duration_sec,
        "frame_count": frame_count,
        "fps": fps,
        "face_presence_ratio": float(face_presence_ratio),

        # raw 行為比例（0~1）
        "eye_contact_ratio": float(eye_contact_ratio),
        "smile_ratio": float(smile_ratio),

        # 高階分數（0~10）
        "eye_contact_score": float(eye_contact_score),
        "facial_positivity_score": float(facial_positivity_score),
        "posture_stability_score": float(posture_stability_score),
        "gesture_expressiveness_score": float(gesture_expressiveness_score),
        "head_nodding_score": float(head_nodding_score),
        "fidgeting_score": float(fidgeting_score),
    }

