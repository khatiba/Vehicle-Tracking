# Vehicle Detection and Tracking

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a Linear Support Vector Machine (SVM) classifier
* Apply a color transform and append binned color features, as well as histograms of color, to the HOG feature vector. 
* Normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use the trained classifier to search for vehicles in images.
* Run pipeline on a video stream (testing with `test_video.mp4` and later implement on full `project_video.mp4`) and create a heat map of recurring detections frame by frame
* Use the heatmap to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/hog.png
[image3]: ./output_images/test1.png
[image4]: ./output_images/test2.png
[image5]: ./output_images/test3.png
[image6]: ./output_images/test4.png
[image7]: ./output_images/test5.png
[image8]: ./output_images/test6.png

[image9]: ./examples/bboxes_and_heat.png
[image10]: ./examples/labels_map.png
[image11]: ./examples/output_bboxes.png
[video12]: ./project_video.mp4


### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the first code cell of the IPython notebook.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).
I selected some random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`. `YCrCb` performed the best results with the least false positives.

![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried many combinations of parameters and trained the classifier with each set, I always achieved over 98% accuracy during training, but the following
parameters for the spatial, color and HOG gave me a final accuracy around 99.2%.

| ----          | ----          |
| Color Space       | 'YCrCb' |
| Orient            | 8 |
| Pix per Cell      | 8 |
| Cell per Block    | 2 |
| Hog Channels      | ALL |
| Spatial Size      | (32, 32) |
| Hist Bins         | 32 |

#### 3. Training a Linear Support Vector Machine (SVM)

I trained a linear SVM by first normalizing the feature vector which composed of color histogram and histogram of oriented gradients. Using scikit learn, I split the data set with 80% for training and 20% for validation. The Linear SVM model was trained in just a few seconds on an AWS GPU instance with a final accuracy of about 99%.


### Sliding Window Search
Searching the images was carried out by first cropping the search area to just the road. Then a HOG feature extraction is taken on the region once per frame. Within the region, HOG features and spatial and color features are extracted, normalized and a prediction is run to see if a car is detected.

In order to improve the precision of bounding box, I divided up the region from top to bottom, smaller windows were confined to the top of the region where cars would appear the smallest. The window size then increased moving towards the foreground where cars appear larger.

Ultimately I searched on three scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result. I did try using only HOG, HOG + Spatial, HOG + Color combinations but they all produced more false positives than all three combined.

![alt text][image3]
![alt text][image4]
![alt text][image5]

---

### Video Implementation

Here is the final video impementation including lane line detection.

[Final Project Video](./project_video.mp4)


#### Detection and False Positives
I recorded the positions of positive detections in each frame of the video. From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions. I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap. I then assumed each blob corresponded to a vehicle. I constructed bounding boxes to cover the area of each blob detected.  

Examples of this are shown above, with the blue boxes representing all detections, the heatmap and then finally the positive detections in green.

![alt text][image6]
![alt text][image7]
![alt text][image8]

---

### Discussion

I tried many combinations of color spaces for HOG feature extraction and color histogram. I found that HOG worked best on the lightness L or Y channels. I tried different orients and pixels per cell for HOG extraction to find values that produced good training results but were general enough to avoid over-fitting.

Spatial and color histogram features weren't the focus of my adjustments, I settled on values that provided good training accuracy but kept the feature size low. There are still some false positives that make it through and the bounding box is constantly adjusting. I did my best to reduce the jutter by banding the search region for each search window size. I think the SVM classifier performed well, I didn't augment the dataset but I think that would have reduced the false positives.
