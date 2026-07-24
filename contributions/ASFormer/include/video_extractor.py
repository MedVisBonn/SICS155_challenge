import cv2
import numpy as np
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed, wait
import multiprocessing


class VideoExtractor:
    def __init__(self, video_path, output_dir, batch_size=8, target_fps=15):
        self.video_path = video_path
        self.batch_size = batch_size
        self.output_dir = output_dir
        self.batch_number = 0
        if not os.path.exists(output_dir):
            os.makedirs(output_dir) 
        self.target_fps = target_fps
        #print(f'{os.path.basename(video_path)} has {self.frame_count} frames with {self.fps} FPS, resulting in {self.frame_count // batch_size} batches.')
        self.buffer = []
        self.cap = cv2.VideoCapture(video_path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        

    def get_next_batch(self):
        #self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        frame_count = 0
        while frame_count < self.total_frames:
            frames = []
            if len(self.buffer) == 0:
                for _ in range(self.batch_size):
                    ret, frame = self.cap.read()
                    if not ret: 
                        return
                    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                self.buffer = frames
            else:
                frames = self.buffer[1:]
                ret, frame = self.cap.read()
                if not ret: 
                    return
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            frame_count += 1
            
            if len(frames) != self.batch_size:
                return
            
            batch_array = np.array(frames)
            #print(batch_array.shape)
            yield batch_array
            self.batch_number += 1
            if self.batch_number % 1000 == 0: 
                print(self.batch_number)

    def save_batch(self, batch_array):
        np.save(os.path.join(self.output_dir, f'batch_{self.batch_number}.npy'), batch_array)
        
    def __del__(self):
        self.cap.release()



def extract_frames(video_path, frames_dir):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(os.path.join(frames_dir, f'frame_{frame_count:05d}.jpg'), frame)
        frame_count += 1
    cap.release()
    return frames_dir


if __name__ == '__main__':
    # Example usage:
    video_path = r'C:\Users\Simon\Data\2024_SICS_Phase\Timestamps annotation\Video 1.mp4'
    batch_size = 64  # Replace N with the desired batch size
    output_dir = 'temp'

    extractor = VideoExtractor(video_path, batch_size, output_dir)
    for batch in extractor.get_next_batch():
        extractor.save_batch(batch)
