import os
import random

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
