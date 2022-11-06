# Turn Prediction for Autonomous Navigation

## Description

Given a video of a dashcam of a car, this project attempts to isolate the curved lane and calculate its curvature to generate navigational commands.


## Data

* Dashcam video of a car

![alt text](/data/data.gif)


## Approach

* Isolate the trapezoidal section containing the immediate portion of the lane

* Apply perspective transform on this trapezoidal section to gain a birds eye view of this lane.

* Convert to grayscale, apply adaptive thresholding, median blurring and thresholding to get a binary warped image.

* Apply a sliding window approach to detect the lanes. The two columns from left and right with the most number of white pixels were selected and these columns were divided into 10 parts.

* Next, horizontal ranges were considered around each of these divisions and the center of all these divisions were found and ellipses drawn around these centers.

* 2 curves were fit on these centers and Inverse perspective transform was applied on this to fit this back into the original frame.

* This trapezoidal section wass highlighted with a mask.

* The distances were converted from pixels to meters and the curvatures of these curves was found.

* Based on the average curvatures, navigational commands ('Turn Left', 'Go Straight', 'Turn Right') were generated.

* To account for cases where the lanes were not visible, navigational commands from previous valid frames were used. This condition was set to be activated when the number of white pixels in the binary image exceeded a certain threshold.

* All of this information was combined and displayed in real time.


## Output

* Binary Warped Image

![alt text](/output/out1.jpg)

* Lane plots highlighted

![alt text](/output/out2.jpg)

* Highlighted lanes on the original frame

![alt text](/output/out3.jpg)

* Output when lane is found

![alt text](/output/out4.jpg)

* Output when lane is not found

![alt text](/output/out5.jpg)

* Final Output [Link] (https://drive.google.com/file/d/1g75wtSCWvYk7_pmEpAO1RM4qIMk4fhZc/view?usp=sharing)

![alt text](/output/outvid.gif)


## Getting Started

### Dependencies

<p align="left"> 
<a href="https://www.python.org" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" alt="python" width="40" height="40"/>&ensp; </a>
<a href="https://numpy.org/" target="_blank" rel="noreferrer"> <img src="https://www.codebykelvin.com/learning/python/data-science/numpy-series/cover-numpy.png" alt="numpy" width="40" height="40"/>&ensp; </a>
<a href="https://opencv.org/" target="_blank" rel="noreferrer"> <img src="https://avatars.githubusercontent.com/u/5009934?v=4&s=400" alt="opencv" width="40" height="40"/>&ensp; </a>

* [Python 3](https://www.python.org/)
* [NumPy](https://numpy.org/)
* [OpenCV](https://opencv.org/)


### Executing program

* Clone the repository into any folder of your choice.
```
git clone https://github.com/ninadharish/AR-Tag-Detection-and-Tracking.git
```

* Open the repository and navigate to the `src` folder.
```
cd AR-Tag-Detection-and-Tracking/src
```
* Depending on whether you want to superimpose athe image or 3D cube on the tag, comment/uncomment the proper line.

* Run the program.
```
python main.py
```


## Authors

👤 **Ninad Harishchandrakar**

* [GitHub](https://github.com/ninadharish)
* [Email](mailto:ninad.harish@gmail.com)
* [LinkedIn](https://linkedin.com/in/ninadharish)
