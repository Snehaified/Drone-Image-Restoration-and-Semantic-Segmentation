# Drone-Image-Restoration-and-Semantic-Segmentation
Developed a multi-modal drone image processing pipeline that leverages auxiliary sensor data to perform image restoration using iterative Richardson–Lucy deconvolution and adaptive contrast normalization followed by object detection.

**Overview**

The goal of this project is to design a custom image recovery and texture-based segmentation pipeline for drone imagery affected by motion blur and artefacts. Using additional sensor data (IMU), the system restores degraded images and segments them into semantic classes such as grass, trees, and cars, followed by object instance detection.

**Pipeline**

The implemented algorithm processes each image through the following steps:

1. Load degraded drone image and motion blur kernel derived from IMU measurements

2. Perform iterative Richardson–Lucy deconvolution in the frequency domain

3. Apply Wallis filtering for local contrast enhancement

4. Extract texture features using Laws filter bank

5. Perform texture-based semantic segmentation

6. Apply majority voting for spatial consistency

7. Detect and count specific objects (e.g., cars) using morphological operations and connected-component analysis

**0utput**
<img width="2206" height="1106" alt="image" src="https://github.com/user-attachments/assets/5140839c-3ecd-47fb-9bef-cebf5fa28528" />


**Tools & Techniques**

- MATLAB

- Frequency-domain deconvolution

- Probabilistic image restoration

- Texture analysis (Laws filters)

- Morphological image processing

