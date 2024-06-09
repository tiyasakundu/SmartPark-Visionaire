
import numpy as np
import cv2
from copy import deepcopy
from PIL import Image
import pytesseract as tess
import numpy as np
import smtplib
import  imutils
import sys
import pytesseract
import pandas as pd
import time
def preprocess(img):
	#cv2.imshow("Input",img)
	imgBlurred = cv2.GaussianBlur(img, (5,5), 0)
	#cv2.imshow("Gaussian blur Image", imgBlurred)
	gray = cv2.cvtColor(imgBlurred, cv2.COLOR_BGR2GRAY)
	#cv2.imshow("BGR TO GRAY", gray)
	sobelx = cv2.Sobel(gray,cv2.CV_8U,1,0,ksize=3)
	#cv2.imshow("Sobel edge detection",sobelx)
	#cv2.waitKey(0)
	ret2,threshold_img = cv2.threshold(sobelx,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	#cv2.imshow("Threshold",threshold_img)
	#cv2.waitKey(0)
	return threshold_img

def cleanPlate(plate):
	print("CLEANING PLATE. . .")
	gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
	#kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
	#thresh= cv2.dilate(gray, kernel, iterations=1)

	_, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
	contours,hierarchy = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

	if contours:
		areas = [cv2.contourArea(c) for c in contours]
		max_index = np.argmax(areas)

		max_cnt = contours[max_index]
		max_cntArea = areas[max_index]
		x,y,w,h = cv2.boundingRect(max_cnt)

		if not ratioCheck(max_cntArea,w,h):
			return plate,None

		cleaned_final = thresh[y:y+h, x:x+w]
		cv2.imshow("Function Test",cleaned_final)
		return cleaned_final,[x,y,w,h]

	else:
		return plate,None


def extract_contours(threshold_img):
	element = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(17, 3))
	morph_img_threshold = threshold_img.copy()
	cv2.morphologyEx(src=threshold_img, op=cv2.MORPH_CLOSE, kernel=element, dst=morph_img_threshold)
	cv2.imshow("Morphed",morph_img_threshold)
	cv2.waitKey(0)

	contours, hierarchy= cv2.findContours(morph_img_threshold,mode=cv2.RETR_EXTERNAL,method=cv2.CHAIN_APPROX_NONE)
	return contours


def ratioCheck(area, width, height):
	ratio = float(width) / float(height)
	if ratio < 1:
		ratio = 1 / ratio

	aspect = 4.7272
	min = 15*aspect*15  # minimum area
	max = 125*aspect*125  # maximum area

	rmin = 3
	rmax = 6

	if (area < min or area > max) or (ratio < rmin or ratio > rmax):
		return False
	return True

def isMaxWhite(plate):
	avg = np.mean(plate)
	if(avg>=115):
		return True
	else:
 		return False

def validateRotationAndRatio(rect):
	(x, y), (width, height), rect_angle = rect

	if(width>height):
		angle = -rect_angle
	else:
		angle = 90 + rect_angle

	if angle>15:
	 	return False

	if height == 0 or width == 0:
		return False

	area = height*width
	if not ratioCheck(area,width,height):
		return False
	else:
		return True



def cleanAndRead(img,contours):
	#count=0
	print(len(contours))
	for i,cnt in enumerate(contours):
		min_rect = cv2.minAreaRect(cnt)
		print(min_rect)

		if validateRotationAndRatio(min_rect):

			x,y,w,h = cv2.boundingRect(cnt)
			plate_img = img[y:y+h,x:x+w]
			print('---------------------------------------------------------')
			print(plate_img)


			if(isMaxWhite(plate_img)):
				#count+=1
				clean_plate, rect = cleanPlate(plate_img)
				print('---------------------------------------------------------')
				print(rect)

				if rect:
					x1,y1,w1,h1 = rect
					x,y,w,h = x+x1,y+y1,w1,h1
					cv2.imshow("Cleaned Plate",clean_plate)
					cv2.waitKey(0)
					plate_im = Image.fromarray(clean_plate)
					text = tess.image_to_string(plate_im, lang='eng')
					print("Detected Text : ",text)
					print("Detected Text : ",text)


					img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
					cv2.imshow("Detected Plate",img)
					cv2.waitKey(0)

	#print "No. of final cont : " , count



if __name__ == '__main__':
	print( "DETECTING PLATE . . .")

	img = cv2.imread("testData/test1.jpg")
	#img = cv2.imread("car.jpeg")

	threshold_img = preprocess(img)
	contours = extract_contours(threshold_img)
	pytesseract.pytesseract.tesseract_cmd = 'C:\Program Files\Tesseract-OCR/tesseract.exe'
	#image = cv2.imread('car.jpeg')

	image = imutils.resize(img, width=500)

	cv2.imshow("Original Image", image)

	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	cv2.imshow("1 - Grayscale Conversion", gray)

	gray = cv2.bilateralFilter(gray, 11, 17, 17)
	cv2.imshow("2 - Bilateral Filter", gray)

	imgBlurred = cv2.GaussianBlur(gray, (5, 5), 0)
	cv2.imshow("3.Gaussian blur Image", imgBlurred)

	sobelx = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=3)
	cv2.imshow("4.Sobel edge detection", sobelx)

	edged = cv2.Canny(gray, 170, 200)
	cv2.imshow("5 - Canny Edges", edged)



	(cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:30]
	NumberPlateCnt = None

	count = 0
	for c in cnts:
		peri = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.02 * peri, True)
		if len(approx) == 4:
			NumberPlateCnt = approx
			break

	# Masking the part other than the number plate
	mask = np.zeros(gray.shape, np.uint8)
	new_image = cv2.drawContours(mask, [NumberPlateCnt], 0, 255, -1)
	new_image = cv2.bitwise_and(image, image, mask=mask)
	edged = cv2.Canny(new_image , 170, 200)
	cv2.imshow("Final Image ", edged)
	cv2.namedWindow("Final_image", cv2.WINDOW_NORMAL)
	cv2.imshow("Final_image", new_image)


	# Configuration for tesseract
	config = ('-l eng --oem 1 --psm 3')

	# Run tesseract OCR on image
	text = pytesseract.image_to_string(new_image, config=config)

	# Data is stored in CSV file
	raw_data = {'date': [time.asctime(time.localtime(time.time()))],
				'v_number': [text]}

	df = pd.DataFrame(raw_data, columns=['date', 'v_number'])
	df.to_csv('data.csv')

	# Print recognized text

	print(text)

	cv2.waitKey(0)
	cleanAndRead(img, contours)

	if len(contours)!=0:
		print( len(contours)) #Test
		cv2.drawContours(img, contours, -1, (0,255,0), 1)
		cv2.imshow("Contours",img)
		cv2.waitKey(0)

