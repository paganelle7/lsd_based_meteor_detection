# Meteor detection using line segment detection algorithm
Here is the code for detecting a meteor from a sequence of images. The program uses the pytorch module. 
The program consists of 3 steps:
# 1. Denoising
All the methods used to prepare the map are based on cadr_processing.py
# 2. Meteor detection
Meteor_segment_detection.py includes functions for getting line segments from an image.  Note, that main metod lsd_optimized give two tensor sequence from two direction of iteration.
# 3 Postprocesiing
At this stage, the line segments form pairs and can be identified as meteors.


