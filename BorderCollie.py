import cv2
import numpy as np

source = cv2.VideoCapture('Border Collies - 7259.mp4')
FourCC = cv2.VideoWriter_fourcc('m','p','4','v')
video = cv2.VideoWriter('Output.mp4',FourCC,25,(1920,1080))

net = cv2.dnn.readNet("yolov3_training_last.weights", "yolov3_testing.cfg")
classes = ["Border Collie"]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

return_value, frame = source.read()
count = 1
while return_value:
	
	height, width, channels = frame.shape
	
	blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
	net.setInput(blob)
	outs = net.forward(output_layers)
	
	class_ids = []
	confidences = []
	boxes = []
	
	for out in outs:
		for detection in out:
			scores = detection[5:]
			class_id = np.argmax(scores)
			confidence = scores[class_id]
			
			if confidence > 0.5:
				center_x = int(detection[0] * width)
				center_y = int(detection[1] * height)
				w = int(detection[2] * width)
				h = int(detection[3] * height)

				x = int(center_x - w / 2)
				y = int(center_y - h / 2)

				boxes.append([x, y, w, h])
				confidences.append(float(confidence))
				class_ids.append(class_id)

	indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
	for i in range(len(boxes)):
		if i in indexes:
			x, y, w, h = boxes[i]
			label = str(classes[class_ids[i]])
			cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 102, 255), 2)
			cv2.putText(frame, label, (x, y - 15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 102, 255), 2)

	video.write(frame)
	print('Finished Processing Frame: ', count)
	count += 1
	return_value, frame = source.read()
	
source.release()
video.release()
cv2.destroyAllWindows()
	
