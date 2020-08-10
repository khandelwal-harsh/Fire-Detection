from fire_detection import FireDetection
from tqdm import tqdm
import cv2


fire = FireDetection()

cap = cv2.VideoCapture('path to input video file')

output_path = 'out_video1.mp4'
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
pbar = tqdm(total=total_frames, desc="[INFO] Processing video")

# Create the VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
count_frames = 0

while cap.isOpened():

	ret,frame = cap.read()
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

	if ret == False:
		break

	output,img = fire.predict(frame,0.8)
	for out in output:
		confidence,label = out
	if label == 'Fire':
	    cv2.putText(frame,label,(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0), 1)
	else:
		cv2.putText(frame,label,(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0), 1)


	frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
	count_frames += 1
	pbar.update(1)
	cv2.imshow("Frame", frame)
	video.write(frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
pbar.close()
cap.release()
video.release()
cv2.destroyAllWindows()

