import cv2

video = cv2.VideoCapture("video.mp4")

total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
frame_rate = video.get(cv2.CAP_PROP_FPS)

extract_frame = int(frame_rate * 5)

for i in range(0, total_frames, extract_frame):
    video.set(cv2.CAP_PROP_POS_FRAMES, i)
    success, frame = video.read()
    
    if success:
        cv2.imwrite("output/frame_{}.jpg".format(i), frame)

video.release()