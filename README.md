# Vehicle Detection and Tracking

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a Linear Support Vector Machine (SVM) classifier
* Apply a color transform and append binned color features, as well as histograms of color, to the HOG feature vector. 
* Normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use the trained classifier to search for vehicles in images.
* Run pipeline on a video stream (testing with `test_video.mp4` and later implement on full `project_video.mp4`) and create a heat map of recurring detections frame by frame
* Use the heatmap to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.


[Video Sample](./video-sample.gif)


[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/hog.png
[image3]: ./output_images/test1.png
[image4]: ./output_images/test2.png
[image5]: ./output_images/test3.png
[image6]: ./output_images/test4.png
[image7]: ./output_images/test5.png
[image8]: ./output_images/test6.png


I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).

I selected some random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

### Histogram of Oriented Gradients (HOG)

The code for this step is contained in the second code cell of the IPython notebook under `Utilities`.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`.

![alt text][image2]

#### Choosing HOG Parameters

I tried many combinations of parameters and trained the classifier with each set and achieved 95%-98% accuracy during trainings. I found that using spatial and color histograms weren't adding any value to the classifier and were slow. I also found that `YUV` color space provided the best results (and least false positives during testing). I continued to tune just the HOG parameters and achieved good training results with the following:

| Parameter         | Value         |
| ----              | ----          |
| Color Space       | YUV           |
| Orient            | 11            |
| Pix per Cell      | 16            |
| Cell per Block    | 2             |
| Hog Channels      | ALL           |


#### Training a Linear Support Vector Machine (SVM)

I trained a linear SVM purely on HOG features. After much testing, I didn't see any added value in spatial or color histograms for this project and they added a lot of overhead. However, they may be useful in different lighting conditions or road conditions.




### Sliding Window Search

Searching the images was carried out by first cropping the search area to just the road. Then a HOG feature extraction is taken on the region once per frame. Within the region, HOG features are extracted from each window and run through the SVM predictor.

In order to improve the precision of the bounding box, I divided up the region from top to bottom, smaller windows were confined to the top of the region where cars would appear the smallest. The window size then increased moving towards the foreground where cars appear larger.

Ultimately I searched on 6 sub regions and scales using `YUV` 3-channel HOG features. To reduce the jitter in the bouding box, I also keep track of the last set of detected bounding boxes. I use these to filter the next frame and increase the heat map region. Finally I optimized the search to achieve nearly 2 frames per second during processing.

![alt text][image3]
![alt text][image4]
![alt text][image5]

---

### Video Implementation

Here is the final video impementation including lane line detection.

[Final Project Video with Lane Detection](./project_video_output.mp4)


#### Detection and False Positives

I recorded the positions of positive detections in each frame of the video. I also track and add the previous detected positions. From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions. I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap. I constructed bounding boxes to cover the area of each blob detected.  

Examples of this are shown above, with the blue boxes representing all detections, the heatmap and then finally the positive detections in green.

![alt text][image6]
![alt text][image7]
![alt text][image8]

---

### Discussion

I tried many combinations of color spaces for HOG feature extraction and color histogram. I found that HOG worked well on the lightness L or Y channels. However, without spatial or color features, HOG on 1 channel resulted in many false positives. I tried different orients and pixels per cell for HOG extraction to find values that produced good training results but were general enough to avoid over-fitting.

Spatial and color histogram features weren't the focus of my adjustments, I found that they weren't adding any significant value to the classifier and just increased processing time. There are still some false positives that make it through and the bounding box is constantly adjusting. I did my best to reduce the jutter by banding the search region for each search window size and using previously detected regions to augment the heatmap.

There is still much improvement that can be done to reduce the jittering of the bounding box. Overall, I think the classifier performed well and the search could be further optimized by augmenting the training data set. 
