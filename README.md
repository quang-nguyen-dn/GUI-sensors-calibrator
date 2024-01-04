# Perception sensors calibration toolbox with web-based GUI

## Install

1. Create a conda environment:
    ```bash
    conda create -n calib python=3.10
    conda activate calib
    ```

2. Install the required packages and dependencies:
    ```bash
    sudo apt update
    sudo apt install -y libegl1 libgl1 libgomp1
    pip install -r requirements.txt
    ```

3. Install MATLAB Python binding in the current conda environment:
    ```bash
    cd path/to/MATLAB/{R2023a}/extern/engines/python
    pip install -e .
    ```

## Input and output data structure

The programme supports both input data in rosbag1 and rosbag2 formats.

```
{DATA_ROOT} (sample_data)
└── {WORKSPACE_NAME} (data_ladybug5)
    ├── rosbags
    │   ├── {rosbag1-no.1}.bag
    │   ├── {...}.bag
    │   ├── {rosbag1-no.n}.bag
    │   │
    │   │── {rosbag2-no.1}
    │   │   └── Rosbag2 no. 1 content files
    │   ├── {rosbag2-...}
    │   └── {rosbag2-no.m}
    │       └── Rosbag2 no. 2 content files
    └── calib
        └── {FIRST_CAM_NAME} (rgb_0)
            ├── intrinsics.json (Camera intrinsics parameters, if available)
            ├── imgs
            │   ├── 0.png
            │   ├── 1.png
            │   ├── 2.png
            ├── {FIRST_LIDAR_NAME} (livox)
            │   ├── 0.npy
            │   ├── 1.npy
            │   ├── 2.npy
            └── {SECOND_LIDAR_NAME} (os128)
                ├── 0.npy
                ├── 1.npy
                ├── 2.npy
                └── extrinsics.json (Generated results after extrinsic calibration is completed)
```


## Usage

1. Activate the conda environment:
    ```bash
    conda activate calib
    ```
2. Run the calibration GUI:
    ```bash
    python gui.py
    ```
3. Follow the on-screen instructions to calibrate your perception sensors.
