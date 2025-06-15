import os
import cv2
import mediapipe as mp
import pandas as pd
import json
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Directories
VIDEO_DIR = r"C:\Users\Idan\Desktop\All\SIGN TALK\divisions\merged"  # Folder where your videos are stored
OUTPUT_DIR = r"C:\Users\Idan\Desktop\All\SIGN TALK\Full dataset\pose landmarks"  # Folder where output .csv, .json, .gif will be saved
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize MediaPipe Holistic
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    refine_face_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Pose connection map
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24), (23, 25), (25, 27), (24, 26),
    (26, 28), (27, 29), (28, 30), (29, 31), (30, 32)
]

POSE_COUNT, FACE_COUNT, HAND_COUNT = 33, 468, 21

def process_video(video_path, video_name):
    cap = cv2.VideoCapture(video_path)
    frame_index = 0
    all_keypoints = []

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(frame_rgb)

        frame_data = [frame_index]

        # Pose
        if results.pose_landmarks:
            for lm in results.pose_landmarks.landmark:
                frame_data.extend([lm.x, lm.y, lm.z, lm.visibility])
        else:
            frame_data.extend([0.0] * (POSE_COUNT * 4))

        # Face
        face_landmarks = results.face_landmarks.landmark if results.face_landmarks else []
        for i in range(FACE_COUNT):
            if i < len(face_landmarks):
                lm = face_landmarks[i]
                frame_data.extend([lm.x, lm.y, lm.z])
            else:
                frame_data.extend([0.0, 0.0, 0.0])

        # Left Hand
        left_landmarks = results.left_hand_landmarks.landmark if results.left_hand_landmarks else []
        for i in range(HAND_COUNT):
            if i < len(left_landmarks):
                lm = left_landmarks[i]
                frame_data.extend([lm.x, lm.y, lm.z])
            else:
                frame_data.extend([0.0, 0.0, 0.0])

        # Right Hand
        right_landmarks = results.right_hand_landmarks.landmark if results.right_hand_landmarks else []
        for i in range(HAND_COUNT):
            if i < len(right_landmarks):
                lm = right_landmarks[i]
                frame_data.extend([lm.x, lm.y, lm.z])
            else:
                frame_data.extend([0.0, 0.0, 0.0])

        all_keypoints.append(frame_data)
        frame_index += 1

    cap.release()

    # Save to CSV
    columns = ['frame']
    columns += [f'pose_{i}_{c}' for i in range(POSE_COUNT) for c in ['x', 'y', 'z', 'vis']]
    columns += [f'face_{i}_{c}' for i in range(FACE_COUNT) for c in ['x', 'y', 'z']]
    columns += [f'left_{i}_{c}' for i in range(HAND_COUNT) for c in ['x', 'y', 'z']]
    columns += [f'right_{i}_{c}' for i in range(HAND_COUNT) for c in ['x', 'y', 'z']]
    
    csv_path = os.path.join(OUTPUT_DIR, f"{video_name}.csv")
    df = pd.DataFrame(all_keypoints, columns=columns)
    df.to_csv(csv_path, index=False)

    # Save to JSON
    frames = []
    for _, row in df.iterrows():
        frame_data = {"frame": int(row["frame"]), "pose": {}, "face": {}, "left": {}, "right": {}}

        for i in range(POSE_COUNT):
            frame_data["pose"][f"{i}"] = {
                "x": row[f"pose_{i}_x"],
                "y": row[f"pose_{i}_y"],
                "z": row[f"pose_{i}_z"],
                "vis": row[f"pose_{i}_vis"]
            }
        for i in range(FACE_COUNT):
            frame_data["face"][f"{i}"] = {
                "x": row[f"face_{i}_x"],
                "y": row[f"face_{i}_y"],
                "z": row[f"face_{i}_z"]
            }
        for i in range(HAND_COUNT):
            frame_data["left"][f"{i}"] = {
                "x": row[f"left_{i}_x"],
                "y": row[f"left_{i}_y"],
                "z": row[f"left_{i}_z"]
            }
            frame_data["right"][f"{i}"] = {
                "x": row[f"right_{i}_x"],
                "y": row[f"right_{i}_y"],
                "z": row[f"right_{i}_z"]
            }

        frames.append(frame_data)

    json_path = os.path.join(OUTPUT_DIR, f"{video_name}.json")
    with open(json_path, "w") as f:
        json.dump(frames, f)

    return frames, fps  # Return for animation step

def create_gif(json_data, gif_path, fps):
    fig, ax = plt.subplots(figsize=(6, 8))
    scat_pose = ax.scatter([], [], s=20, color='blue')
    lines_pose = [ax.plot([], [], 'orange')[0] for _ in POSE_CONNECTIONS]
    scat_face = ax.scatter([], [], s=1, color='gray')
    scat_left = ax.scatter([], [], s=10, color='green')
    scat_right = ax.scatter([], [], s=10, color='red')
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)
    ax.axis('off')

    def update(frame_idx):
        frame = json_data[frame_idx]
        pose_x = [frame["pose"][str(i)]["x"] for i in range(33)]
        pose_y = [frame["pose"][str(i)]["y"] for i in range(33)]
        scat_pose.set_offsets(list(zip(pose_x, pose_y)))
        for idx, (p1, p2) in enumerate(POSE_CONNECTIONS):
            lines_pose[idx].set_data([pose_x[p1], pose_x[p2]], [pose_y[p1], pose_y[p2]])

        face_x = [frame["face"][str(i)]["x"] for i in range(468)]
        face_y = [frame["face"][str(i)]["y"] for i in range(468)]
        scat_face.set_offsets(list(zip(face_x, face_y)))

        left_x = [frame["left"][str(i)]["x"] for i in range(21)]
        left_y = [frame["left"][str(i)]["y"] for i in range(21)]
        scat_left.set_offsets(list(zip(left_x, left_y)))

        right_x = [frame["right"][str(i)]["x"] for i in range(21)]
        right_y = [frame["right"][str(i)]["y"] for i in range(21)]
        scat_right.set_offsets(list(zip(right_x, right_y)))

        return [scat_pose, scat_face, scat_left, scat_right] + lines_pose

    ani = animation.FuncAnimation(fig, update, frames=len(json_data), interval=1000 / fps, blit=True)
    ani.save(gif_path, writer="pillow", fps=fps)
    plt.close(fig)

# Process all videos in the folder
for file in os.listdir(VIDEO_DIR):
    if file.endswith(".mp4"):
        video_name = os.path.splitext(file)[0]
        video_path = os.path.join(VIDEO_DIR, file)
        print(f"🔄 Processing: {video_name}")
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        json_data, _ = process_video(video_path, video_name)
        gif_path = os.path.join(OUTPUT_DIR, f"{video_name}.gif")
        create_gif(json_data, gif_path, fps)

        print(f"✅ Done: {video_name}.gif")
