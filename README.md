# DropleX

Welcome to the ***DropleX*** repository! This project contains a Tracking Pipeline that utilizes StarDist for object detection and classification and implements an Unscented Kalman filter with RTS smoother to further suppress measurement error. Moreover, this project contains an Analysis Pipeline that processes the resulting trajectories in a window-framework to accommodate out of equilibrium dynamics.

If you want to run the tracking pipeline and analyze of the resulting trajectories using the provided sample data, follow the instructions below. 

## Getting Started

Follow the steps below to set up and run the project on your local machine.

---

### Prerequisites

Ensure you have the following software installed:

- [Python 3.x](https://www.python.org/downloads/) (required)
- [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install) (for package management)

### Installation

1. **Clone the Repository**
   Clone the DropleX repository to your local machine:
   ```bash
   git clone https://github.com/skandiz/DropleX
   ```
2. **Navigate to the Project Directory**  
   Move into the project folder:
   ```bash
   cd DropleX
   ```
3. **Install Dependencies**  
   Install all required Python libraries:
   ```bash
   conda env create -f environment.yml
   ```

4. **Activate conda environment**
   Activate the DropleX conda environment:
   ```bash
   conda activate DropleX
   ```

5. **Download sample data**
   Download sample video and trajectories:
   ```bash
   python3 setup.py
   ```


### Running the Tracking pipeline

To execute the Tracking pipeline, run the following command:
```bash
python3 main_tracking.py
```

### Running the Analysis

To execute the Tracking pipeline, run the following command:
```bash
python3 main_analysis.py
```

---

## How to use DropleX tracking pipeline

### Interactive Workflow

Once the application starts, you will be guided through an interactive menu:

1. **Select Video**  
   Choose a the video to track. Videos are loaded from the `video_input` folder:
   ```bash
   🤔 Which video do you want to track?
      1. Video1
      2. Video2
      ...
   ```

2. **Select Stardist Model**  
   Choose a Stardist model for detection & classification. Models are loaded from the `stardist_models` folder:
   ```bash
   🤔 Which Stardist model do you want to use?
      1. model1
      2. model2
      ...
   ```
3. **Choose Tracking Pipeline part**
   Select the part(s) of the tracking pipeline you want to run:
   ```bash
   🤔 Which part do you want to run?
      1. Test
      2. Detection & Classification
      3. Linking
      4. Interpolation
      5. Kalman filter & RTS Smoother
      6. All of them
   ```
4. **Choose Interpolation Kernel**
   If Interpolation is enabled, select the kernel to use:
   ```bash
   🤔 Which kernel do you want to use for interpolation?
      1. linear
      2. nearest
      3. quadratic
      4. cubic
   ```

5. **Choose Run Mode**
   Select either to run the process or import the data:
   ```bash
   🤔 Do you want to run the tracking or import the data?
      1. Import the data
      2. Run the tracking
   ```
6. **Save Plots**  
   Specify whether to save plots during the tracking:
   ```bash
   🤔 Do you want to save the plots during the tracking? (yes/no)
   ```

7. **Animated Plots**  
   Specify whether to process animated plots during the tracking:
   ```bash
   🤔 Do you want to plot animations during the tracking? (yes/no)
   ```

8. **Show Plots**  
   Specify whether to show plots during the tracking:
   ```bash
   🤔 Do you want to see the plots during the tracking? (yes/no)
   ```

9. **Review Your Choices**
   The program will recap your selections and ask for confirmation:
   ```bash
   --------------------------------------- RECAP ---------------------------------------
         Video selection:                  sample_video
         Model name:                       skandiz_model_rgb
         Resolution:                       1000x1000 px
         Test:                             Enabled
         detection:                        Enabled
         linking:                          Enabled
         Interpolation:                    Enabled
         Interpolation method:             linear
         Kalman filter & RTS smoother:     Enabled


         Save plots:                       Enabled
         Show plots:                       Enabled
         Animated plots:                   Enabled
   -------------------------------------------------------------------------------------

   🤔 Do you want to proceed with these choices? (yes/no)
   ```

### Output

Results will be saved in the `tracking_results/{video}` directory.

The directory structure for output will look like this:
```bash
output/
├── video1/
│   ├── plot1.png
│   ├── raw_detection.parquet
│   ├── raw_tracking.parquet
│   ├── interpolated_tracking.parquet
│   ├── kalman_rts_trajectories.parquet
│   └── ...
├── video2/
│   └── ...
```

### Training the Stardist deep neural network




---

## How to use DropleX analysis

### Interactive Workflow

Once the application starts, you will be guided through an interactive menu:

1. **Select Trajectories**  
   Choose a the trajectory for analysis. Trajectories available are listed from the analysis_config.json file:
   ```bash
   🤔 Which trajectory do you want to analyze?
      1. trajectory1
      2. trajectory2
      ...
   ```

2. **Choose Analysis Type**  
   Select the type(s) of analysis you want to run:
   ```bash
   🤔 Which analyses do you want to run? (1, 2, ...)
      1. Order parameters analysis
      2. Shape analysis
      3. Time Averaged Mean Squared Displacement analysis
      4. Speed distribution analysis
      5. Turning angles distribution analysis
      6. Velocity Autocovariance analysis
      7. Dimer distribution analysis
      8. All of them
   ```

3. **Choose Run Mode**
   Select either to run the process or import the data:
   ```bash
   🤔 Do you want to run the analysis or import the data?
      1. Import the data
      2. Run the analysis
   ```
4. **Save Plots**  
   Specify whether to save plots during the analysis:
   ```bash
   🤔 Do you want to save the plots during the analysis? (yes/no)
   ```

5. **Animated Plots**  
   Specify whether to process animated plots during the analysis:
   ```bash
   🤔 Do you want to plot animations during the analysis? (yes/no)
   ```

6. **Show Plots**  
   Specify whether to show plots during the analysis:
   ```bash
   🤔 Do you want to see the plots during the analysis? (yes/no)
   ```

7. **Review Your Choices**  
   The program will recap your selections and ask for confirmation:
   ```bash
   Video has 36000 frames with a resolution of 1080x1080 and a framerate of 30.000833356482126 fps
   --------------------------------------- RECAP ---------------------------------------
         Video selection:                  sample_video
         Order analysis:                   Enabled
         Shape analysis:                   Enabled
         TAMSD analysis:                   Enabled
         Speed analysis:                   Enabled
         Turning angles analysis:          Enabled
         Velocity Autocovariance analysis: Enabled
         Dimer distribution analysis:      Enabled


         Number of particles: 50
         The trajectory has 36000 frames at 30 fps --> 1200.00 s
         Windowed analysis: windows of 100 s and stride of 10 s --> 110 steps
         The evolution is divided in the following stages:
            Stage 1 starts at: 0h 0m 0s
            Stage 2 starts at: 0h 4m 0s
            Stage 3 starts at: 0h 8m 0s
            Stage 4 starts at: 0h 12m 0s
            Stage 5 starts at: 0h 16m 0s
   -------------------------------------------------------------------------------------
   🤔 Do you want to proceed with these choices? (yes/no)
   ```


### Output of the trajectory analysis

Results will be saved in the `output/{video_name}` directory, organized by analysis type. If enabled, visual plots will also be generated.

The directory structure for output will look like this:
```bash
output/
├── video_name1/
│   ├── order_analysis/
│   │   ├── plot1.png
│   │   └── plot2.png
│   ├── shape_analysis/
│   │   ├── plot1.png
│   │   └── plot2.png
│   ├── tamsd_analysis/
│   │   ├── plot1.png
│   │   └── plot2.png
│   └── ...
├── video_name2/
│   └── ...
```