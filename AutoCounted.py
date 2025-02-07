import cv2
import numpy as np
import easyocr
import imutils
import matplotlib.pyplot as plt

image = cv2.imread('Images/ImageCar6.jpg') #Path to image 
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
imgfilter = cv2.bilateralFilter(gray, 11, 15, 15)
edges = cv2.Canny(imgfilter, 30, 200)

cont = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cont = imutils.grab_contours(cont)
cont = sorted(cont, key = cv2.contourArea, reverse=True)[:8]
pos = None
for con in cont:
    approx = cv2.approxPolyDP(con, 10, True)
    if len(approx) == 4:
        pos = approx
        break

mask = np.zeros(gray.shape, np.uint8)
new_ing = cv2.drawContours(mask,[pos], 0, 255, -1)
bitvice_img = cv2.bitwise_and(image, image, mask = mask)

(x, y) = np.where(mask == 255)
(x1, y1) = (np.min(x), np.min(y))
(x2, y2) = (np.max(x), np.max(y))
crop = gray[x1:x2, y1:y2]
text = easyocr.Reader(['en'])
text = text.readtext(crop)
res = (text[0][-2])
label = cv2.putText(image,res,(x1, y2), cv2.FONT_HERSHEY_PLAIN,3, (0, 0, 255),1)

final_image = cv2.rectangle(image,(y1, x1), (y2, x2),(0,255,0),2)
print(res)
plt.imshow(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB))


plt.show()