import os
import random
from moviepy import VideoFileClip, concatenate_videoclips



#search videos on one label and sentence level
def search_video_by_label(directory, label):
    label = str(label)
    """
    Searches for video files in the directory with names starting with the given label
    followed by A, B, C, D, E, or F.
    Randomly returns one of the matching video file paths, or None if none found.
    """
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
    valid_suffixes = ('A', 'B', 'C', 'D', 'E', 'F')
    matches = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(video_extensions):
                file_name_without_ext = os.path.splitext(file)[0]
                if (file_name_without_ext.startswith(label) and
                    len(file_name_without_ext) > len(label) and
                    file_name_without_ext[len(label)] in valid_suffixes):
                    matches.append(os.path.join(root, file))

    if matches:
        return random.choice(matches)
    
    return None



#search video on words (could be multiple ids in labels array)
def search_word_videos_by_labels(directory, labels):
    """
    For each label, tries to find a corresponding video file (exact match without extension).
    Returns a list of matching video file paths, in the same order as labels.
    """
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
    label_to_file = {}


    # First, index all available video files
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(video_extensions):
                file_name_without_ext = os.path.splitext(file)[0]
                label_to_file[file_name_without_ext] = os.path.join(root, file)

    # Now, retrieve in labels order
    matches = []
    for label in labels:
        file_path = label_to_file.get(str(label))
        matches.append(file_path)  # Can be None if not found

    return matches


def concatenate_videos(video_paths, output_path):
    clips = []
    for path in video_paths:
        if path and os.path.exists(path):
            clip = VideoFileClip(path)
            clips.append(clip)
        else:
            print(f"⚠️ Video not found: {path}")

    if clips:
        final_clip = concatenate_videoclips(clips, method="compose")  # 'compose' handles size/fps differences
        final_clip.write_videofile(output_path, codec='libx264')
        print(f"✅ Merged video saved to: {output_path}")
        return output_path
    else:
        print("❌ No videos to merge.")
        return None



def search_word_gifs_by_labels(directory, labels):
    """
    For each label, tries to find a corresponding GIF file (exact match without extension).
    Returns a list of matching GIF file paths, in the same order as labels.
    """
    if not isinstance(labels, (list, tuple)):
        labels = [labels]

    gif_extension = ('.gif',)
    label_to_file = {}

    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(gif_extension):
                file_name_without_ext = os.path.splitext(file)[0]
                label_to_file[file_name_without_ext] = os.path.join(root, file)

    matches = []
    for label in labels:
        file_path = label_to_file.get(str(label))
        matches.append(file_path)

    return matches





# def get_or_create_merged_video(labels, video_paths, media_root_path):
#     sentence_csv_path = media_root_path / "word_to_sentence_map.csv"
#     os.makedirs(sentence_csv_path.parent, exist_ok=True)

#     # Construct sentence string
#     sentence = " ".join(labels)

#     # Check if sentence exists in CSV
#     existing_id = None
#     if os.path.exists(sentence_csv_path):
#         with open(sentence_csv_path, mode='r', newline='', encoding='utf-8') as file:
#             reader = csv.reader(file)
#             for row in reader:
#                 if len(row) >= 2 and row[1] == sentence:
#                     existing_id = row[0]
#                     break

#     if existing_id:
#         print(f"✅ Existing merged video found for sentence: '{sentence}' with ID: {existing_id}")
#         existing_video_path = media_root_path / "merged_sentences" / f"{existing_id}.mp4"
#         return str(existing_video_path), existing_id

#     # If not, merge videos
#     print(f"🔄 Merging new video for sentence: '{sentence}'")

#     # Determine next ID
#     if os.path.exists(sentence_csv_path):
#         with open(sentence_csv_path, mode='r', newline='', encoding='utf-8') as file:
#             reader = csv.reader(file)
#             ids = [int(row[0]) for row in reader if row and row[0].isdigit()]
#             next_id = max(ids, default=0) + 1
#     else:
#         next_id = 1

#     output_video_path = media_root_path / "merged_words_videos" / f"{next_id}.mp4"

#     # Concatenate videos
#     clips = []
#     for path in video_paths:
#         if path and os.path.exists(path):
#             clip = VideoFileClip(path)
#             clips.append(clip)
#         else:
#             print(f"⚠️ Video not found: {path}")

#     if clips:
#         final_clip = concatenate_videoclips(clips, method="compose")
#         final_clip.write_videofile(str(output_video_path), codec='libx264')
#         print(f"✅ Merged video saved to: {output_video_path}")

#         # Append mapping to CSV
#         with open(sentence_csv_path, mode='a', newline='', encoding='utf-8') as file:
#             writer = csv.writer(file)
#             writer.writerow([next_id, sentence])

#         return str(output_video_path), next_id
#     else:
#         print("❌ No videos to merge.")
#         return None, None

