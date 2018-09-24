import sys
import cv2
import math
import os
import numpy as np
import multiprocessing as mp

# File Paths
video_path = './input_video.avi'
raw_frame_path = './raw/'
processed_frame_path = './processed/'
path_to_output_video = './processed_video.mov'

# Multiprocessing config
core_count = int(mp.cpu_count())

# Macro block size, radius, threshold
k = 9
radius = 1
threshold = 200

# Video properties
frame_height = 0
frame_width = 0
frame_rate = 0
total_frames = 0

# Manual video properties config
# frame_height = 720
# frame_width = 1280
# frame_rate = 30
# total_frames = 300

# Extract video frames
def extract_frames():
    cap = cv2.VideoCapture(video_path)
    frame_counter = 0   

    global frame_height, frame_width, frame_rate
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))

    while(1):
        reture_flag, frame = cap.read()
        if not reture_flag:
            print('All frames extracted')
            break
        
        cv2.imwrite(raw_frame_path + 'frame%d.tif' % frame_counter, frame)
        frame_counter += 1

        if cv2.waitKey(30) & 0xff == ord('q'):
            break
    cap.release()
    global total_frames
    total_frames = frame_counter

# Draw directional arrows of motion field
def arrowdraw(img, x1, y1, x2, y2):
    radians = math.atan2(x1 - x2, y2 - y1)
    x11 = 0
    y11 = 0
    x12 = -10
    y12 = -10

    u11 = 0
    v11 = 0
    u12 = 10
    v12 = -10

    x11_ = x11 * math.cos(radians) - y11 * math.sin(radians) + x2
    y11_ = x11 * math.sin(radians) + y11 * math.cos(radians) + y2

    x12_ = x12 * math.cos(radians) - y12 * math.sin(radians) + x2
    y12_ = x12 * math.sin(radians) + y12 * math.cos(radians) + y2

    u11_ = u11 * math.cos(radians) - v11 * math.sin(radians) + x2
    v11_ = u11 * math.sin(radians) + v11 * math.cos(radians) + y2

    u12_ = u12 * math.cos(radians) - v12 * math.sin(radians) + x2
    v12_ = u12 * math.sin(radians) + v12 * math.cos(radians) + y2 

    img = cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 1)
    # img = cv2.circle(img, (x1, y1), 2, (204, 0, 102), 1)
    # img = cv2.circle(img, (x2, y2), 2, (51, 255, 51), 1)

    img = cv2.line(img, (int(x11_), int(y11_ )), (int(x12_), int(y12_)), (255 , 0, 0), 1) 
    img = cv2.line(img, (int(u11_), int(v11_ )), (int(u12_), int(v12_)), (255 , 0, 0), 1)

    return img

# Worker for each process
def process_frames_worker(start, end):
    for frame_counter in range(start, end, 1):
        current_frame = cv2.imread(raw_frame_path + 'frame%d.tif' % frame_counter)
        img = current_frame
        next_frame = cv2.imread(raw_frame_path + 'frame%d.tif' % (frame_counter + 1))

        if current_frame is None or next_frame is None:
            break

        # Convert images to numpy array
        current_frame = np.array(cv2.imread(raw_frame_path + 'frame%d.tif' % frame_counter), np.int64)
        next_frame = np.array(cv2.imread(raw_frame_path + 'frame%d.tif' % (frame_counter + 1)), np.int64)

        frame_width = int(current_frame.shape[1])
        frame_height = int(current_frame.shape[0])

        for col in range(0, current_frame.shape[1], k):
            for row in range(0, current_frame.shape[0], k):
                
                x = 0
                y = 0
                min_ssd = math.inf
                
                x_start = col - k * radius
                x_fin = col + k + k * radius

                y_start = row - k * radius
                y_fin = row + k + k * radius

                if x_start < 0:
                    x_start = 0
                if x_fin > frame_width:
                    x_fin = frame_width
                if y_start < 0:
                    y_start = 0
                if y_fin > frame_height:
                    y_fin = frame_height

                # Iterate through blocks to search
                for c in range(x_start, x_fin, k):
                    for r in range(y_start, y_fin, k):

                        # Calculate SSD
                        ssd = math.sqrt(np.sum(np.square(current_frame[row:row + k, col:col + k] - next_frame[r:r + k, c:c + k])))
               
                        if ssd < min_ssd:
                            # y = r + math.floor(k / 2)
                            # x = c + math.floor(k / 2)
                            y = r
                            x = c
                            min_ssd = ssd
                
                # Draw arrows
                if (y != row) and (x != col) and (min_ssd > threshold):
                    img = arrowdraw(img, col + math.floor(k / 2), row + math.floor(k / 2), x + math.floor(k / 2), y + math.floor(k / 2))

        print("Frame: %d complete" % frame_counter)

        # Output processed frame
        cv2.imwrite(processed_frame_path + 'frame%d.tif' % frame_counter, img)

# Process frames using multiple processes for speedup
def process_frames():
    # Count total frames
    total_frames = len(os.listdir(raw_frame_path))

    # List of processes
    process_list = []

    # Start processes
    for i in range(core_count):
        print("Thread: %d started" % i)
        start = int(i * (total_frames / core_count))
        end = int((total_frames / core_count) * (i + 1))
        if i == core_count - 1:
            end = total_frames
        process = mp.Process(target=process_frames_worker, args=(start, end))
        process_list.append(process)
        process.start()
        
    # Join processes
    for process in process_list:
        process.join()
    
    print("All frames processed")

# Export processed frames as a .mov             
def stitch_frames():
    out = cv2.VideoWriter(path_to_output_video, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), frame_rate, (int(frame_width), int(frame_height)))
    frame_counter = 0
    while(1):
        img = cv2.imread(processed_frame_path + 'frame%d.tif' % frame_counter)
        if img is None:
            print('Video export complete')
            break
        out.write(img)
        frame_counter += 1
    out.release()

if __name__ == '__main__':
    # create output folders if they don't exist
    if not os.path.exists(raw_frame_path):
        os.makedirs(raw_frame_path)
    if not os.path.exists(processed_frame_path):
        os.makedirs(processed_frame_path)
    
    extract_frames()

    print("Video Properties")
    print("Width: %d" % frame_width)
    print("Height: %d" % frame_height)
    print("Frame rate: %d" % frame_rate)
    print("Total frames: %d" % total_frames)

    process_frames()
    stitch_frames()
