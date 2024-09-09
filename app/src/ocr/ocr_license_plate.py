from skimage.io import imread
import imutils
import cv2

from app.src.ocr.anpr import PyImageSearchANPR


def cleanup_text(text):
	return "".join([c if ord(c) < 128 else "" for c in text]).strip()

anpr = PyImageSearchANPR(debug=False)

def main_ocr(img, debug=False):
	image = imread(img)
	image = imutils.resize(image, width=600)
	(lpText, lpCnt) = anpr.find_and_ocr(image)

	if debug:
		if lpText is not None and lpCnt is not None:
			box = cv2.boxPoints(cv2.minAreaRect(lpCnt))
			box = box.astype("int")
			cv2.drawContours(image, [box], -1, (0, 255, 0), 2)
			(x, y, w, h) = cv2.boundingRect(lpCnt)
			cv2.putText(image, cleanup_text(lpText), (x, y - 15),
						cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
			print("[INFO] {}".format(lpText))
			cv2.imshow("Output ANPR", image)
			cv2.waitKey(-1)

	return lpText


if __name__ == "__main__":
	img = "license_plates/05.jpg"
	result = main_ocr(img, debug=False)
	print(result)
