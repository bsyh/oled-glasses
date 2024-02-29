# USAGE
# python detect_faces_video.py --prototxt deploy.prototxt.txt --model res10_300x300_ssd_iter_140000.caffemodel

# import the necessary packages
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
from PIL import Image
import requests
from io import BytesIO
import numpy as np

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
prototxt = "deploy.prototxt.txt"
model = "res10_300x300_ssd_iter_140000.caffemodel"
confidence_arg = 0.5
args = vars(ap.parse_args())
# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(prototxt, model)
# loop over the frames from the video stream
def detect_face(frame):
	
	# frame = imutils.resize(frame, width=400)
 
	# grab the frame dimensions and convert it to a blob
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
		(300, 300), (104.0, 177.0, 123.0))
 
	# pass the blob through the network and obtain the detections and
	# predictions
	net.setInput(blob)
	detections = net.forward()

	# loop over the detections
	lst = []
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with the
		# prediction
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the `confidence` is
		# greater than the minimum confidence
		if confidence < confidence_arg:
			continue

		# compute the (x, y)-coordinates of the bounding box for the
		# object
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")
 
		# draw the bounding box of the face along with the associated
		# probability
		# text = "{:.2f}%".format(confidence * 100)
		# y = startY - 10 if startY - 10 > 10 else startY + 10
		# cv2.rectangle(frame, (startX, startY), (endX, endY),
		# 	(0, 0, 255), 2)
		# frame = cv2.putText(frame, str(((startX+endX)/2-frame.shape[1]/2)), (startX, y),
		# 	cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
		lst.append((startY, endY, startX, endX))
	# return frame
	mid_person = lst[0]
	for i in lst:
		if (((i[2]+i[3])/2 - frame.shape[1]/2)**2) < (((mid_person[2]+mid_person[3])/2 - frame.shape[1]/2)**2):
			mid_person = i
	return frame[mid_person[0] : mid_person[1], mid_person[2]: mid_person[3]]
	

if __name__ == '__main__':
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	# r = requests.get("https://encrypted-tbn2.gstatic.com/licensed-image?q=tbn:ANd9GcSfcB96rkysCCHgQJd2l_RzFnat8AkW8MYEum8DTLCU5n9p-eSvRsRlrpk1K_6JgdofrpTZ__fJa_4Vkyo")
	r = requests.get("https://img.freepik.com/free-photo/people-taking-selfie-together-registration-day_23-2149096795.jpg")

	frame = Image.open(BytesIO(r.content))
	frame = np.array(frame)
	# Convert RGB to BGR
	frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
	frame = detect_face(frame)

	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(0) & 0xFF