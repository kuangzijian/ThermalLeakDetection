import torch
import numpy as np 
import os
from PIL import Image
import torchvision.transforms as transforms
from custom_conv_lstm import STAE
import cv2

root_dir = '../../dataset/LeakAI dataset 2023-12-20-flattened/618/618-50-01-0038-01/3/2023-12-20/'
root_dir = '../../dataset/600-50-01-0405-01-flattened/1/2024-04-11/'

transform = transforms.Compose([
                transforms.Resize((100, 100)),
                transforms.Grayscale(),
                transforms.ToTensor()
            ])

def mean_squared_loss(x1, x2):
	diff = x1 - x2
	b, l, ch, h, w = diff.shape
	n_samples = b * l * ch * h * w
	sq_diff = diff ** 2
	dist = np.sqrt(sq_diff.sum())
	# print(dist, n_samples)
	# mean_dist = dist / n_samples

	return dist

threshold=20

model = torch.load("best.pt", map_location=torch.device('cpu'))
model.eval()

cumulative_index = 0

frames = sorted([os.path.join(root_dir, frame) for frame in os.listdir(root_dir) if frame.endswith(".jpg")])
video_length = len(frames)
clips = []
for start_idx in range(0, video_length):
	data = torch.zeros((10, 1, 100, 100), dtype=torch.float32)
	end_idx = min(start_idx + 10, video_length)
	clip_frames = frames[start_idx:end_idx]
	if len(clip_frames) < 10:
		clip_frames += [None] * (10 - len(clip_frames))  # Pad with None
	for i, frame in enumerate(clip_frames):
		if frame is not None:
			frame = Image.open(frame).convert('L')  # Convert to grayscale
			frame = transform(frame)
			data[i] = frame
	clips.append(data)

clips = torch.stack(clips)

for i, bunch in enumerate(clips):
	cur_frame = frames[i]
	frame_numpy = cv2.imread(cur_frame)
	H, W, _ = frame_numpy.shape
	n_bunch=bunch.unsqueeze(0)
	reconstructed_bunch=model(n_bunch)

	loss=mean_squared_loss(n_bunch,reconstructed_bunch.detach().numpy())
	print(loss)
	if loss>threshold:
		print("Anomalous bunch of frames at bunch number {}".format(i))
		flag=1
		cv2.putText(frame_numpy, f"Anomalous bunch of frames at bunch number {i}", (int(W*0.3), 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
		cv2.putText(frame_numpy, f"{loss}", (int(W*0.3), 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
	cv2.imshow('Detected Changes', frame_numpy)

	if cv2.waitKey(100) & 0xFF == ord('q'):
		break


	else:
		print('Bunch Normal')



if flag==1:
	print("Anomalous Events detected")