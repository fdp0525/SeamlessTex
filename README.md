# SeamlessTex
This repository is the implementation of "Seamless Texture Optimization for RGB-D Reconstruction". 
> Yanping Fu, Qingan Yan, Jie Liao, Huajian Zhou, Jin Tang, Chunxia Xiao. <i>Texture Mapping for 3D Reconstruction with RGB-D Sensor</i>



# How to use

## 1. Run
To test our algorithm. run G2LTex in command line:
```
./bin/SeamlessTex [DIR] [PLY] 
```
Params explanation:
-`PLY`: The reconstructed model for texture mapping.
-`DIR`: The texture image directory, include rgb images, depth images, and camera trajectory.

The parameters of the camera and the system can be set in the config file.
```
Config/config.yml
```

## 2. Input Format
- Color frames (color_XX.jpg): RGB, 24-bit, JPG.
- Depth frames (depth_XX.png): depth (mm), 16-bit, PNG (invalid depth is set to 0).
- Camera poses (color_XX.cam): world-to-camera [tx, ty, tz, R00, R01, R02, R10, R11, R12, R20, R21, R22].


## 3. Dependencies
The code has following prerequisites:
- OpenCV (2.4.10)
- Eigen (>3.0)
- png12
- jpeg

## 4. Parameters
All the parameters can be set in the file ```Config/config.yml``` as follows:
```
%YAML:1.0
depth_fx: 540.69
depth_fy: 540.69
depth_cx: 479.75
depth_cy: 269.75
depth_width: 960
depth_height: 540

RGB_fx: 1081.37
RGB_fy: 1081.37
RGB_cx: 959.5
RGB_cy: 539.5
RGB_width: 1920
RGB_height: 1080
.
