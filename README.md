# mobile-document-scanner

OpenCV and Python versions:
This example will run on Python 2.7/3+ and OpenCV 2.4/3+

How To Build a Kick-Ass Mobile Document Scanner in Just 5 Minutes



Whenever you need to perform a 4 point perspective transform, you should be using this module.

And you guessed it, we’ll be using it to build our very own document scanner.

So let’s get down to business.

Open up your favorite Python IDE, (I like Sublime Text 2), create a new file, name it scan.py , and let’s get started.

Building a Document Scanner App using Python, OpenCV, and Computer Vision
# import the necessary packages
from pyimagesearch.transform import four_point_transform
from skimage.filters import threshold_local
import numpy as np
import argparse
import cv2
import imutils
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True,
	help = "Path to the image to be scanned")
args = vars(ap.parse_args())
Lines 2-7 handle importing the necessary Python packages that we’ll need.

We’ll start by importing our four_point_transform  function which I discussed last week.

We’ll also be using the imutils  module, which contains convenience functions for resizing, rotating, and cropping images. You can read more about imutils  in my this post. To install imutils , simply:

Building a Document Scanner App using Python, OpenCV, and Computer Vision
$ pip install --upgrade imutils
Next up, let’s import the threshold_local  function from scikit-image. This function will help us obtain the “black and white” feel to our scanned image.

Note (15 January 2018): The threshold_adaptive  function has been deprecated. This post has been updated to make use of threshold_local .

Lastly, we’ll use NumPy for numerical processing, argparse  for parsing command line arguments, and cv2  for our OpenCV bindings.

Lines 10-13 handle parsing our command line arguments. We’ll need only a single switch image, --image , which is the path to the image that contains the document we want to scan.

Now that we have the path to our image, we can move on to Step 1: Edge Detection.

Step 1: Edge Detection
The first step to building our document scanner app using OpenCV is to perform edge detection. Let’s take a look:

Building a Document Scanner App using Python, OpenCV, and Computer Vision
# load the image and compute the ratio of the old height
# to the new height, clone it, and resize it
image = cv2.imread(args["image"])
ratio = image.shape[0] / 500.0
orig = image.copy()
image = imutils.resize(image, height = 500)
# convert the image to grayscale, blur it, and find edges
# in the image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(gray, 75, 200)
# show the original image and the edge detected image
print("STEP 1: Edge Detection")
cv2.imshow("Image", image)
cv2.imshow("Edged", edged)
cv2.waitKey(0)
cv2.destroyAllWindows()
First, we load our image off disk on Line 17.

In order to speedup image processing, as well as make our edge detection step more accurate, we resize our scanned image to have a height of 500 pixels on Lines 17-20.

We also take special care to keep track of the ratio  of the original height of the image to the new height (Line 18) — this will allow us to perform the scan on the original image rather than the resized image.

From there, we convert the image from RGB to grayscale on Line 24, perform Gaussian blurring to remove high frequency noise (aiding in contour detection in Step 2), and perform Canny edge detection on Line 26.

The output of Step 1 is then shown on Lines 30 and 31.

Take a look below at the example document:

Figure 1: The first step of building a document scanning app. On the left we have the original image and on the right we have the edges detected in the image.
Figure 1: The first step of building a document scanning app. On the left we have the original image and on the right we have the edges detected in the image.
On the left you can see my receipt from Whole Foods. Notice how the picture is captured at an angle. It is definitely not a 90-degree, top-down view of the page. Furthermore, there is also my desk in the image. Certainly this is not a “scan” of any means. We have our work cut out for us.

However, on the right you can see the image after performing edge detection. We can clearly see the outline of the receipt.

Not a bad start.

Let’s move on to Step 2.

Step 2: Finding Contours
Contour detection doesn’t have to be hard.

In fact, when building a document scanner, you actually have a serious advantage…

Take a second to consider what we’re actually building.

A document scanner simply scans in a piece of paper.

A piece of paper is assumed to be a rectangle.

And a rectangle has four edges.

Therefore, we can create a simple heuristic to help us build our document scanner.

The heuristic goes something like this: we’ll assume that the largest contour in the image with exactly four points is our piece of paper to be scanned.

This is also a reasonably safe assumption — the scanner app simply assumes that the document you want to scan is the main focus of our image. And it’s also safe to assume (or at least should be) that the piece of paper has four edges.

And that’s exactly what the code below does:

Building a Document Scanner App using Python, OpenCV, and Computer Vision
# find the contours in the edged image, keeping only the
# largest ones, and initialize the screen contour
cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]
# loop over the contours
for c in cnts:
	# approximate the contour
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.02 * peri, True)
	# if our approximated contour has four points, then we
	# can assume that we have found our screen
	if len(approx) == 4:
		screenCnt = approx
		break
# show the contour (outline) of the piece of paper
print("STEP 2: Find contours of paper")
cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
cv2.imshow("Outline", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
We start off by finding the contours in our edged image on Line 37. We also handle the fact that OpenCV 2.4, OpenCV 3, and OpenCV 4 return contours differently on Line 38.

A neat performance hack that I like to do is actually sort the contours by area and keep only the largest ones (Line 39). This allows us to only examine the largest of the contours, discarding the rest.

We then start looping over the contours on Line 42 and approximate the number of points on Line 44 and 45.

If the approximated contour has four points (Line 49), we assume that we have found the document in the image.

And again, this is a fairly safe assumption. The scanner app will assume that (1) the document to be scanned is the main focus of the image and (2) the document is rectangular, and thus will have four distinct edges.

From there, Lines 55 and 56 display the contours of the document we went to scan.

And now let’s take a look at our example image:

Figure 2: The second step of building a document scanning app is to utilize the edges in the image to find the contours of the piece of paper.
Figure 2: The second step of building a document scanning app is to utilize the edges in the image to find the contours of the piece of paper.
As you can see, we have successfully utilized the edge detected image to find the contour (outline) of the document, illustrated by the green rectangle surrounding my receipt.

Lastly, let’s move on to Step 3, which will be a snap using my four_point_transform function.

Step 3: Apply a Perspective Transform & Threshold
The last step in building a mobile document scanner is to take the four points representing the outline of the document and apply a perspective transform to obtain a top-down, “birds eye view” of the image.

Let’s take a look:

Building a Document Scanner App using Python, OpenCV, and Computer Vision
# apply the four point transform to obtain a top-down
# view of the original image
warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
# convert the warped image to grayscale, then threshold it
# to give it that 'black and white' paper effect
warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
T = threshold_local(warped, 11, offset = 10, method = "gaussian")
warped = (warped > T).astype("uint8") * 255
# show the original and scanned images
print("STEP 3: Apply perspective transform")
cv2.imshow("Original", imutils.resize(orig, height = 650))
cv2.imshow("Scanned", imutils.resize(warped, height = 650))
cv2.waitKey(0)
Line 62 performs the warping transformation. In fact, all the heavy lifting is handled by the four_point_transform  function. Again, you can read more about this function in last week’s post.

We’ll pass two arguments into four_point_transform : the first is our original image we loaded off disk (not the resized one), and the second argument is the contour representing the document, multiplied by the resized ratio.

So, you may be wondering, why are we multiplying by the resized ratio?

We multiply by the resized ratio because we performed edge detection and found contours on the resized image of height=500 pixels.

However, we want to perform the scan on the original image, not the resized image, thus we multiply the contour points by the resized ratio.

To obtain the black and white feel to the image, we then take the warped image, convert it to grayscale and apply adaptive thresholding on Lines 66-68.

Finally, we display our output on Lines 72-74.

Python + OpenCV document scanning results
And speaking of output, take a look at our example document by  running the script:

Building a Document Scanner App using Python, OpenCV, and Computer Vision
$ python scan.py --image images/receipt.jpg
