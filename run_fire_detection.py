from fire_detection import FireDetection
from tqdm import tqdm
import cv2
import numpy

class Fire_Detection(FireDetection):

	def __init__(self):
		super().__init__()

	def run_fire_detection(self,input_file,input_type):

		if input_type == 'image':
			frame = input_file
			frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			
			output,img = super().predict(frame,0.8)
			for out in output:
				confidence,label = out
			print(label)
			if label == 'Fire':
				cv2.putText(frame,label,(50,55),cv2.FONT_HERSHEY_SIMPLEX,3,(0,0,255), 2)
			else:
				cv2.putText(frame,label,(50,55),cv2.FONT_HERSHEY_SIMPLEX,3,(0,255,0), 2)
			output_path = 'output_image.png'
			cv2.imwrite(output_path,frame)
			return output_path

		elif input_type == 'video':

			cap = cv2.VideoCapture(input_file)

			output_path = 'out_video.mp4'
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


				if ret == False:
					break
				frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

				output,img = super().predict(frame,0.8)
				for out in output:
					confidence,label = out
				if label == 'Fire':
					cv2.putText(frame,label,(50,55),cv2.FONT_HERSHEY_SIMPLEX,3,(255,0,0), 2)
				else:
					cv2.putText(frame,label,(50,55),cv2.FONT_HERSHEY_SIMPLEX,3,(0,255,0), 2)



				frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
				count_frames += 1
				pbar.update(1)
				# cv2.imshow("Frame", frame)
				video.write(frame)
				# if cv2.waitKey(1) & 0xFF == ord('q'):
				# 	break
			pbar.close()
			cap.release()
			video.release()
			cv2.destroyAllWindows()
			return output_path

