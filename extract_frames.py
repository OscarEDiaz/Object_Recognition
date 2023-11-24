# @ Oscar Emiliano Ramírez Díaz
# Main code to detect objects each n frames in a static or live video
import numpy as np
import os, shutil
import cv2

from yolov5.detect import run as run_model

CONFIG = {
    'frame_rate': 4,
    'video_read_url': 'video/',
    'result_url': 'result/',
    'raw_frames_save_url': 'raw/',
    'processed_frames_save_url': 'processed/'
}

def get_saving_frames_duration(cap, framerate):
    spots = []

    clip_duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)

    for i in np.arange(0, clip_duration, 1 / framerate):
        spots.append(i)

    return spots

def cut_frames(video, video_frame_rate, saving_frame_rate, saving_folder):
    # Retrieve the time spots of the video to save as frames
    spots = get_saving_frames_duration(video, saving_frame_rate)
    count = 0

    while True:
        is_read, frame = video.read()
        if not is_read:
            # break out of the loop if there are no frames to read
            break
        # get the duration by dividing the frame count by the FPS
        frame_duration = count / video_frame_rate
        try:
            # get the earliest duration to save
            closest_duration = spots[0]
        except IndexError:
            # the list is empty, all duration frames were saved
            break
        if frame_duration >= closest_duration:
            path = os.path.join(saving_folder, f"frame-{count}.jpg")
            cv2.imwrite(path, frame) 
            try:
                spots.pop(0)
            except IndexError:
                pass
        # increment the frame count
        count += 1

def pipeline(video_dir_path):
    # Iterate over the videos folder and process each one
    print(video_dir_path)

    # For each video save the raw frames in its corresponding folder
    for video_url in os.listdir(video_dir_path):
        is_first_iteration = True

        if video_url.endswith('.mp4'):
            saving_folder = os.path.join(CONFIG['raw_frames_save_url'], video_url).replace('.mp4', '')

            if os.path.exists(saving_folder) and is_first_iteration:
                shutil.rmtree(saving_folder)
                is_first_iteration = False
            
            if not os.path.exists(saving_folder):
                os.mkdir(saving_folder)
            
            current_video_dir_path = os.path.join(video_dir_path, video_url)

            video = cv2.VideoCapture(current_video_dir_path)
            video_fps = video.get(cv2.CAP_PROP_FPS)

            # If framerate is above the current video fps get the maximum frame rate
            saving_frame_rate = min(CONFIG['frame_rate'], video_fps)

            # Get the path where the frames of the video were saved
            cut_frames(video, video_fps, saving_frame_rate, saving_folder)

    # For each folder of raw frames proccess them
    for frame_dir_url in os.listdir(CONFIG['raw_frames_save_url']):
        if os.path.isdir(os.path.join(CONFIG['raw_frames_save_url'], frame_dir_url)):
            for frame_url in os.listdir(os.path.join(CONFIG['raw_frames_save_url'], frame_dir_url)):
                if frame_url.endswith('.jpg'):
                    saving_folder = CONFIG['processed_frames_save_url']

                    frame_abs_url = os.path.join(os.path.join(CONFIG['raw_frames_save_url'], frame_dir_url), frame_url)

                    if not os.path.exists(saving_folder):
                        os.mkdir(saving_folder)
                    
                    run_model(source=frame_abs_url, project=saving_folder, name=frame_dir_url)


def main():
    cwd = os.getcwd()

    result_url = CONFIG['result_url']
    raw_frames_url = CONFIG['raw_frames_save_url']
    processed_frames_url = CONFIG['processed_frames_save_url']

    result_dir_path = os.path.join(cwd, result_url)

    if os.path.exists(result_dir_path):
        shutil.rmtree(result_dir_path)

    os.mkdir(result_dir_path)
    
    raw_frames_dir_path = os.path.join(result_dir_path, raw_frames_url)

    if os.path.exists(raw_frames_dir_path):
        shutil.rmtree(raw_frames_dir_path)

    os.mkdir(raw_frames_dir_path)

    processed_frames_dir_path = os.path.join(result_dir_path, processed_frames_url)

    if os.path.exists(processed_frames_dir_path):
        shutil.rmtree(processed_frames_dir_path)

    os.mkdir(processed_frames_dir_path)

    # Update paths in config object
    CONFIG['raw_frames_save_url'] = raw_frames_dir_path
    CONFIG['processed_frames_save_url'] = processed_frames_dir_path

    # Videos directory path
    videos_dir_path = os.path.join(cwd, CONFIG['video_read_url'])

    pipeline(videos_dir_path)

main()