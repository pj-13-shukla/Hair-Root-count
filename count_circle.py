import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('Circle.jpg')

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect circles using Hough Circle Transform
circles = cv2.HoughCircles(
    gray_image,  # Use the grayscale image here
    cv2.HOUGH_GRADIENT,
    dp=1,
    minDist=20,
    param1=50,
    param2=30,
    minRadius=0,
    maxRadius=0
)

if circles is not None:
    circles = np.uint16(np.around(circles))

if circles is not None:
    for circle in circles[0, :]:
        center = (circle[0], circle[1])
        cv2.circle(image, center, 1, (0, 100, 100), 3)
        radius = circle[2]
        cv2.circle(image, center, radius, (255, 0, 255), 3)

    circle_count = len(circles[0])
else:
    circle_count = 0

print(f"Number of circles found: {circle_count}")

plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
