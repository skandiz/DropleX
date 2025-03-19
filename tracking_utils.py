import os
import re 
import json
import numpy as np
import pandas as pd
import matplotlib as mpl
from stardist.models import StarDist2D
from stardist import random_label_cmap
cmap = random_label_cmap()    
import cv2
from PIL import Image
from tqdm import tqdm
tqdm.pandas()
import random
from csbdeep.utils import normalize
import skimage
import trackpy as tp
from filterpy.common import Saver
from filterpy.kalman import MerweScaledSigmaPoints, UnscentedKalmanFilter
import joblib
import multiprocessing
n_jobs = int(multiprocessing.cpu_count()*0.8)
parallel = joblib.Parallel(n_jobs=n_jobs, backend='threading', verbose=0)

def user_message(message, message_type):
    """
    Display a standardized message to the user with optional emojis to make the communications more enjoyable.
    """
    emoji_map = {
        "question": "🤔",
        "saving_graph": "📶",
        "saving_stats": "💾",
        "info": "❗",
        "error": "❌",
        "success": "✅"
    }
    emoji = emoji_map.get(message_type, "ℹ️")
    if emoji == "success" or emoji == "question":
        print(f"\n{emoji} {message}")
    else:
        print(f"{emoji}  {message}")
        
def ask_options(prompt, options):
    """
    Ask the user to select one or more options for the type of analysis.
    """
    print(f"🤔 {prompt}")
    
    for i, option in enumerate(options):
        print(f"   {i+1}. {option}")

    while True:
        try:
            reply = input("➡️  ").replace(" ", "")  # Remove spaces for cleaner input
            selected = {int(i) for i in reply.split(",")}  # Convert to a set of integers

            if selected.issubset(set(range(1, len(options) + 1))):
                return list(selected)  # Convert back to list and return
            else:
                raise ValueError
        except ValueError:
            user_message("Invalid input. Please select valid options (e.g., 1,3,5).", "error")
            
            
def ask_yesno(prompt):
    """
    Ask the user a yes/no question
    """
    valid_yes = {'y', 'Y', 'yes', 'YES', 'Yes'}
    valid_no = {'n', 'N', 'no', 'NO', 'No'}

    print(f"🤔 {prompt} (yes/no)")
    while True:
        option = input("➡️  ").strip()
        if option in valid_yes:
            return True
        elif option in valid_no:
            return False
        else:
            user_message("Invalid input. Please respond with 'yes' or 'no'.", "error")


def print_recap_tracking(choices, model_name, resolution, interp_method):
    """
    Print a recap of the user's choices at the end of the analysis.
    """
    print("   --------------------------------------- RECAP ---------------------------------------")
    print(f"         Video selection:                  {choices.get('video_selection', 'N/A')}")
    print(f"         Model name:                       {model_name}")
    print(f"         Resolution:                       {resolution}x{resolution} px")      
    print(f"         Test:                             {'Enabled' if choices.get('test') else 'Disabled'}")
    print(f"         detection:                        {'Enabled' if choices.get('detect') else 'Disabled'}")
    print(f"         linking:                          {'Enabled' if choices.get('link') else 'Disabled'}")
    print(f"         Interpolation:                    {'Enabled' if choices.get('interpolation') else 'Disabled'}")
    print(f"         Interpolation method:             {interp_method}")
    print(f"         Kalman filter & RTS smoother:     {'Enabled' if choices.get('kalman') else 'Disabled'}")

    print("\n")
    print(f"         Save plots:                       {'Enabled' if choices.get('save') else 'Disabled'}")
    print(f"         Show plots:                       {'Enabled' if choices.get('show') else 'Disabled'}") 
    print(f"         Animated plots:                   {'Enabled' if choices.get('animated') else 'Disabled'}")

    print("   -------------------------------------------------------------------------------------")


def onClick(event):
    global anim_running
    if anim_running:
        ani.event_source.stop()
        anim_running = False
    else:
        ani.event_source.start()
        anim_running = True

def get_video_parameters(video_selection, config_path="./tracking_config.json"):
    """
    Get video parameters from the JSON configuration file.
    
    Parameters
    ----------
    video_selection : str
        Name of the video selection.
    config_path : str, optional
        Path to the JSON configuration file (default: "./video_config.json").
    
    Returns
    -------
    dict
        Dictionary containing the video parameters.
    
    Raises
    ------
    ValueError
        If the video selection is not found in the configuration file.
    """
    try:
        with open(config_path, "r") as file:
            video_config = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        raise RuntimeError(f"Error loading video configuration: {e}")

    config = video_config.get(video_selection)
    if not config:
        raise ValueError(f"Unknown video selection: {video_selection}")

    # Return dictionary 
    return {
        "system_name": config["system_name"],
        "xmin": config["xmin"], "ymin": config["ymin"], "xmax": config["xmax"], "ymax": config["ymax"],
        "petri_diameter": config["petri_diameter"],
        "all_frames": np.arange(*config["all_frames"]),
        "analysis_frames": np.arange(*config["analysis_frames"]),
        "n_instances": config["n_instances"],
        "video_source_path": config["video_source_path"],
        "crop_verb": config["crop_verb"]
    }



def get_n_errors(df, n_instances, frames):
    """
    Get number of errors in the dataframe
    
    Parameters
    ----------
    df : pandas DataFrame
        DataFrame to analyze
    n_instances : int
        Number of instances expected per frame
    frames : list
        List of frames to analyze
    
    Returns
    -------
    tuple of pandas DataFrame, numpy array, numpy array, numpy array, int
        Tuple containing counts per frame, error frames, error frames with more instances, error frames with less instances, max number of consecutive errors
    """
    
    counts_per_frame = df['frame'].value_counts().reindex(frames, fill_value=0).sort_index()
    counts_per_frame = pd.DataFrame({'frame': frames, 'counts': counts_per_frame.values})
    
    err_frames = counts_per_frame[counts_per_frame['counts'] != n_instances]['frame'].values
    err_up = counts_per_frame[counts_per_frame['counts'] > n_instances]['frame'].values
    err_sub = counts_per_frame[counts_per_frame['counts'] < n_instances]['frame'].values

    print(f'Number of errors: {len(err_frames)}/{len(frames)} ({len(err_frames)/len(frames)*100:.2f}%)')
    print(f'Number of errors > {n_instances}: {len(err_up)}/{len(frames)} ({len(err_up)/len(frames)*100:.2f}%)')
    print(f'Number of errors < {n_instances}: {len(err_sub)}/{len(frames)} ({len(err_sub)/len(frames)*100:.2f}%)')

    max_n_of_consecutive_errs = max((len(group) for group in np.split(err_sub, np.where(np.diff(err_sub) > 1)[0] + 1)), default=0)

    print(f'Max number of consecutive errors: {max_n_of_consecutive_errs}')
    
    return counts_per_frame, err_frames, err_up, err_sub, max_n_of_consecutive_errs


# Function to extract frame1 and frame2 from filenames
def extract_frames(filename):
    """
    Extract frame1 and frame2 from the filename
    
    Parameters
    ----------
    filename : str
        Filename to extract frames from
    
    Returns
    -------
    tuple
        Tuple of frame1 and frame2
    """
    
    match = re.search(r'_(\d+)_(\d+)', filename)
    if match:
        frame1, frame2 = map(int, match.groups())
        return frame1, frame2
    return None, None


def concat_dataframes(path, analysis_frames):
    """
    Concatenate all parquet files in the folder and save the concatenated dataframe
    
    Parameters
    ----------
    path : str
        Path to the folder containing the parquet files
    analysis_frames : list
        List of frames to be analyzed
        
        
    Returns
    -------
    int
        0 if successful
    
    """

    # Get a list of all parquet files in the folder
    parquet_files = [f for f in os.listdir(path) if f.endswith('.parquet')]
    # Create a list of tuples (file, frame1, frame2) for sorting
    files_with_frames = [(f, *extract_frames(f)) for f in parquet_files]
    # Sort files based on frame1 (and optionally frame2 if needed)
    sorted_files = sorted(files_with_frames, key=lambda x: (x[1], x[2]))
    # Initialize an empty list to store the dataframes
    dfs = []
    # Loop through sorted files and read them into dataframes
    for file_info in sorted_files:
        parquet_file = file_info[0]
        file_path = os.path.join(path, parquet_file)
        df = pd.read_parquet(file_path)
        dfs.append(df)
    # Concatenate all dataframes
    full_df = pd.concat(dfs, ignore_index=True)
    full_df.reset_index(drop=True, inplace=True)
    # Save the concatenated dataframe
    full_df.to_parquet(f"{path}raw_detection_{analysis_frames[0]}_{analysis_frames[-1]}.parquet")
    return 0

def draw_polygons(polygons, points, scores, colors, alpha, show_dist, ax):
    """
    Draw polygons and points on the image
    
    Parameters
    ----------
    polygons : list
        List of polygons
    points : list
        List of points
    scores : list
        List of scores
    colors : list
        List of colors
    alpha : float
        Alpha value for transparency
    show_dist : bool
        Whether to show distance lines
    ax : matplotlib axis
        Axis to draw on
    
    Returns
    -------
    matplotlib axis
        Axis with polygons and points drawn
    """
    
    if colors is None:
        cmap = random_label_cmap(len(polygons)+1)
        colors = cmap.colors[1:]
        
    for point, poly, score, c in zip(points, polygons, scores, colors):
        if point is not None:
            ax.plot(point[1], point[0], '.', markersize=8*score, color=c, alpha = alpha)

        if show_dist:
            dist_lines = np.empty((poly.shape[-1],2,2))
            dist_lines[:,0,0] = poly[1]
            dist_lines[:,0,1] = poly[0]
            dist_lines[:,1,0] = point[1]
            dist_lines[:,1,1] = point[0]
            ax.add_collection(mpl.collections.LineCollection(dist_lines, colors=c, linewidths=0.1, alpha = alpha))

        a,b = list(poly[1]), list(poly[0])
        a += a[:1]
        b += b[:1]
        ax.plot(a,b,'--', linewidth=3*score, zorder=1, color=c, alpha = alpha)
    return ax

# kalman filter transition and measure functions

def fx2(x, dt):
    """
    State transition function for a second order motion model.
    
    Parameters
    ----------
    x : numpy array
        State vector
    dt : float
        Time step
    
    Returns
    -------
    numpy array
        Updated state vector
    """
    
    F = np.array([[1, dt, 0,  0],
                  [0, 1,  0,  0],
                  [0, 0,  1, dt],
                  [0, 0,  0,  1]], dtype=float)
    return np.dot(F, x)

def hx2(x):
    """
    Measurement function for a second order motion model.
    
    Parameters
    ----------
    x : numpy array
        State vector
    
    Returns
    -------
    numpy array
        Measurement vector
    """
    
    return np.array([x[0], x[2]])



def fx3(x, dt):
    """
    State transition function for a third order motion model.
    
    Parameters
    ----------
    x : numpy array
        State vector
    dt : float
        Time step
    
    Returns
    -------
    numpy array
        Updated state vector
    """
    
    F = np.array([[1, dt, dt**2/2, 0,  0,       0],
                  [0,  1,      dt, 0,  0,       0],
                  [0,  0,       1, 0,  0,       0],
                  [0,  0,       0, 1, dt, dt**2/2],
                  [0,  0,       0, 0,  1,      dt],
                  [0,  0,       0, 0,  0,       1]], dtype = float)
    return np.dot(F, x)

def hx3(x):
    """
    Measurement function for a third order motion model.
    
    Parameters
    ----------
    x : numpy array
        State vector
    
    Returns
    -------
    numpy array
        Measurement vector
    """
    
    return np.array([x[0], x[3]])


class TrackingVideo:
    """
    Class to track droplets in a video
    
    Attributes
    ----------
    system_name : str
        Name of the system
    model_name : str
        Name of the model
    model : StarDist2D
        StarDist2D model
    video : cv2.VideoCapture
        Video capture object
    fps : float
        Frames per second
    xmin : int
        Minimum x-coordinate
    ymin : int
        Minimum y-coordinate
    xmax : int
        Maximum x-coordinate
    ymax : int
        Maximum y-coordinate
    resolution : int
        Resolution
    w : int
        Width of the video
    h : int
        Height of the video
    n_frames_video : int
        Number of frames in the video
    frames : list
        List of frames
    n_instances : int
        Number of instances
    res_path : str
        Path to save results
    interp_method : str
        Interpolation method    
    
    Methods
    -------
    get_frame(frame, crop_verb, resize_verb)
        Get a frame from the video
    detect_instances_frame(instance_properties, frame, img, full_details)
        Detect instances in a frame
    detect_instances(analysis_frames, crop_verb, resize_verb, full_details)
        Detect instances in the video
    linking_detection(df, cutoff, max_frame_gap, min_trajectory_length)
        Link detections
    interp_raw_trajectory(raw_trajectory_df)
        Interpolate trajectories
    unscented_kalman_filter(measurements, r0, v0, a0, a, b, measurement_sigma, order, cov_factor, q, subsample_factor, saver_verb)
        Unscented Kalman Filter
    kalman_filter_full(trajectories, a, b, measurement_sigma, order, cov_factor, q, subsample_factor)
        Kalman filter
    """
    
    
    def __init__(self, system_name, model_name, video_source_path, xmin, ymin, xmax, ymax, resolution, n_instances, interp_method, res_path):
        self.system_name = system_name
        self.model_name = model_name
        print(f'Loading model {self.model_name}')
        if self.model_name not in ['2D_versatile_fluo', '2D_paper_dsb2018', '2D_versatile_he']:
            self.model = StarDist2D(None, name = model_name, basedir = './stardist_models/')
        else:
            self.model = StarDist2D.from_pretrained(model_name)

        if self.model.config.n_channel_in == 1:
            self.gray_scale_verb = True
        else:
            self.gray_scale_verb = False

        self.video = cv2.VideoCapture(video_source_path)
        self.fps = self.video.get(cv2.CAP_PROP_FPS)
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.resolution = resolution
        self.w = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.n_frames_video = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f'Video has {self.n_frames_video} frames with a resolution of {self.w}x{self.h} and a framerate of {self.fps} fps')
        self.frames = [i for i in range(int(self.video.get(cv2.CAP_PROP_FRAME_COUNT)))]
        self.n_instances = n_instances
        self.res_path = res_path
        self.interp_method = interp_method
        
        
    def get_frame(self, frame, crop_verb, resize_verb):
        """
        Get a frame from the video
        
        Parameters
        ----------
        frame : int
            Frame number
        crop_verb : bool
            Whether to crop the image
        resize_verb : bool
            Whether to resize the image
        Returns
        -------
        numpy array
            Frame
        """
        
        self.video.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ret, image = self.video.read()
        image = np.array(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        if crop_verb:
            image = image[self.ymin:self.ymax, self.xmin:self.xmax]
            
        if resize_verb:
            if image.shape[0] > 1000: # if the image is too large --> shrinking with INTER_AREA interpolation
                image = cv2.resize(image, (self.resolution, self.resolution), interpolation = cv2.INTER_AREA)
            else: # if the image is too small --> enlarging with INTER_LINEAR interpolation
                image = cv2.resize(image, (self.resolution, self.resolution), interpolation = cv2.INTER_CUBIC)
                
        if self.gray_scale_verb:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return image
    
    
    def detect_instances_frame(self, instance_properties, frame, img, full_details):
        """
        Detect instances in a frame
        
        Parameters
        ----------
        instance_properties : dict
            Dictionary to store instance properties
        frame : int
            Frame number
        img : numpy array
            Frame
        full_details : bool
            Whether to include full details
        
        Returns
        -------
        dict
            Dictionary containing instance properties
        """
        
        # Predict instances in the image
        segmented_image, details = self.model.predict_instances(
            normalize(img, 1, 99.8, axis = (0, 1)), predict_kwargs={'verbose': False}
        )
        
        # Extract properties from the labeled image
        instance_properties_frame = skimage.measure.regionprops_table(
            label_image=segmented_image, intensity_image=img,
            properties = (
                'centroid', 'area', 'eccentricity',
                'axis_major_length', 'axis_minor_length', 'orientation',
                'inertia_tensor', 'intensity_mean'
            )
        )

        # Append values of the current frame
        for key, values in instance_properties_frame.items():
            instance_properties[key].extend(values)
            
        num_instances = len(instance_properties_frame['centroid-0'])
        instance_properties['frame'].extend([frame] * num_instances)
        instance_properties['prob'].extend(details['prob'])
        
        # Include class_id and class_prob if model is multi-class
        if self.model.config.n_classes is not None:
            instance_properties['class_id'].extend(details['class_id'])
            instance_properties['class_prob'].extend(details['class_prob'])

        # Include full details if required
        if full_details:
            instance_properties['coord'].extend(details['coord'])
            instance_properties['points'].extend(details['points'])

        return instance_properties


    def detect_instances(self, analysis_frames, crop_verb, resize_verb, full_details):
        """
        Detect instances in the video
        
        Parameters
        ----------
        analysis_frames : list
            List of frames to analyze
        crop_verb : bool
            Whether to crop the image
        resize_verb : bool
            Whether to resize the image
        full_details : bool 
            Whether to include full details
        
        Returns
        -------
        pandas DataFrame
            DataFrame containing instance properties
        """
        
        
        # Initialize dictionary to store instance properties
        # Change as needed (change properties in detect_instances_frame function accordingly)
        base_keys = [
            'centroid-0', 'centroid-1', 'area', 'eccentricity',
            'axis_major_length', 'axis_minor_length', 'orientation',
            'inertia_tensor-0-0', 'inertia_tensor-0-1', 'inertia_tensor-1-0', 'inertia_tensor-1-1',
            'frame', 'prob'
        ]
        
        # Include dict keys if model is multi-class
        if self.model.config.n_classes is not None:
            base_keys.extend(['class_id', 'class_prob'])
        
        if self.model.config.n_channel_in == 1:
            base_keys.extend(['intensity_mean'])          
        else:
            base_keys.extend(['intensity_mean-0', 'intensity_mean-1', 'intensity_mean-2'])
        
        # Include full details keys if required
        if full_details:
            base_keys.extend(['coord', 'points'])

        results_dict = {key: [] for key in base_keys}

        # Process frame
        for frame in tqdm(analysis_frames, desc = "Detecting instances from video"):
            img = self.get_frame(frame, crop_verb, resize_verb)
            self.detect_instances_frame(results_dict, frame, img, full_details)

        # Convert dictionary to DataFrame
        df = pd.DataFrame(results_dict)
        df = df.rename(columns={'centroid-0': 'y', 'centroid-1': 'x'})
        df['frame'] = df['frame'].astype(int)

        # Compute instance radius from area (assuming circular shape)
        df['r'] = np.sqrt(df['area'] / np.pi)
        
        # Sort dataframe by frame and probability
        df = df.sort_values(by=['frame', 'prob'], ascending=[True, False])

        return df

    def linking_detection(self, df, cutoff, max_frame_gap, min_trajectory_length):
        """
        Link detections
        
        Parameters
        ----------  
        df : pandas DataFrame
            DataFrame containing raw detections
        cutoff : float
            Cutoff distance
        max_frame_gap : int
            Maximum frame gap
        min_trajectory_length : int
            Minimum trajectory length
            
        Returns
        -------
        pandas DataFrame
            DataFrame containing linked trajectories
        """
        
        raw_trajectory_df = tp.link(f = df, search_range = cutoff, memory = max_frame_gap, link_strategy = 'hybrid', neighbor_strategy = 'KDTree', adaptive_stop = 1)
        raw_trajectory_df = raw_trajectory_df.sort_values(['frame', 'particle'])
        raw_trajectory_df = tp.filter_stubs(raw_trajectory_df, min_trajectory_length)

        # CREATE COLOR COLUMN AND SAVE DF
        n = len(raw_trajectory_df.particle.unique())
        print(f'N of droplets: {n}')
        random.seed(5)
        colors = ['#'+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(n)]
        for i in range(max(raw_trajectory_df.particle)+1-n):
            colors.append('#00FFFF')
            
        c = []
        
        for p in raw_trajectory_df.particle:
            c.append(colors[p])
        raw_trajectory_df['color'] = c
        raw_trajectory_df = raw_trajectory_df.reset_index(drop=True)        
        return raw_trajectory_df
    
    def interpolate_dataframe(self, group):
        interp_method = self.interp_method
        all_frames = pd.DataFrame({"frame": range(group["frame"].min(), group["frame"].max() + 1)})
        merged = pd.merge(all_frames, group, on="frame", how="left")
        merged = merged.sort_values(by="frame")
        properties = merged.columns.difference(["frame", "particle", "color"])
        for col in properties:
            merged[col] = merged[col].interpolate(method=interp_method)
        # ffill() --> Fill NA/NaN values by propagating the last valid observation to next valid.
        merged["particle"] = merged["particle"].ffill()
        merged["color"]    = merged["color"].ffill()
        return merged
    
    def interp_raw_trajectory(self, raw_trajectory_df):
        """
        Interpolate trajectories
        
        Parameters
        ----------
        raw_trajectory_df : pandas DataFrame
            DataFrame containing raw trajectories
    
        Returns
        -------
        pandas DataFrame
            DataFrame containing interpolated trajectories
        """
        
        if self.interp_method is None:
            raise ValueError('You must set the interp_method parameter first')
        
        temp = raw_trajectory_df.particle
        interpolated_trajectory_df = raw_trajectory_df.copy()
        interpolated_trajectory_df = raw_trajectory_df.groupby('particle').progress_apply(self.interpolate_dataframe)
        interpolated_trajectory_df['particle'] = interpolated_trajectory_df['particle'].astype(int)
        interpolated_trajectory_df = interpolated_trajectory_df.reset_index(drop = True)
        interpolated_trajectory_df = interpolated_trajectory_df.sort_values(['frame', 'particle'])
        interpolated_trajectory_df = interpolated_trajectory_df.reset_index(drop = True)        
        return interpolated_trajectory_df


    def unscented_kalman_filter(self, measurements, r0, v0, a0, a, b, measurement_sigma, order, cov_factor, q, subsample_factor, saver_verb):
        """
        Unscented Kalman Filter
        
        Parameters
        ----------
        measurements : numpy array
            Measurements
        r0 : numpy array
            Initial position
        v0 : numpy array
            Initial velocity
        a0 : numpy array
            Initial acceleration
        a : float
            Alpha value
        b : float   
            Beta value
        measurement_sigma : float
            Measurement sigma
        order : int
            Order
        cov_factor : float
            Covariance factor
        q : float
            Process noise
        subsample_factor : int
            Subsample factor
        saver_verb : bool
            Whether to save the results
        
        Returns
        -------
        tuple
            Tuple containing mu, cov, xs, Ps, s    
        """
        
        dt = 1/int(self.fps/subsample_factor)
        if order == 2:
            # sigma points
            points = MerweScaledSigmaPoints(4, alpha = a, beta = b, kappa = 3 - 4)
            # Unscented Kalman Filter
            ukf = UnscentedKalmanFilter(dim_x = 4, dim_z = 2, dt = dt, fx = fx2, hx = hx2, points = points)
            # initial state estimate : x, vx, ax, y, vy, ay
            ukf.x = np.array([r0[0], v0[0], r0[1], v0[1]])
            # initial covariance matrix
            ukf.P = np.eye(4) * cov_factor
            # process noise covariance matrix for second order kalman filter
            ukf.Q = q**2 * np.array([[dt**4/4, dt**3/2,       0,       0],
                                     [dt**3/2,   dt**2,       0,       0],
                                     [      0,       0, dt**4/4, dt**3/2],
                                     [      0,       0, dt**3/2,   dt**2]], dtype = float)
            
        elif order == 3:
            points = MerweScaledSigmaPoints(6, alpha=a, beta=b, kappa = 3 - 6)
            ukf = UnscentedKalmanFilter(dim_x=6, dim_z=2, dt=dt, fx=fx3, hx=hx3, points=points)
            # initial state estimate : x, vx, ax, y, vy, ay
            ukf.x = np.array([r0[0], v0[0], a0[0], r0[1], v0[1], a0[1]])
            # initial covariance matrix
            ukf.P = np.eye(6) * cov_factor
            # process noise covariance matrix for third order kalman filter
            ukf.Q = q**2 * np.array([[dt*+6/36, dt**5/12,  dt**4/6,              0,        0,        0],
                                     [dt**5/12,  dt**4/4,  dt**3/2,              0,        0,        0],
                                     [ dt**4/6,  dt**3/2,    dt**2,              0,        0,        0],
                                     [       0,        0,        0,       dt**6/36, dt**5/12,  dt**4/6],
                                     [       0,        0,        0,       dt**5/12,  dt**4/4,  dt**3/2],
                                     [       0,        0,        0,       dt**4/6,   dt**3/2,    dt**2]], dtype = float)
        else:
            raise ValueError("Order must be 2 or 3")
        
        if saver_verb:
            s = Saver(ukf)
        # measurement noise covariance matrix
        ukf.R = np.eye(2) * measurement_sigma**2
        
        # perform kalman filter
        if saver_verb:
            mu, cov = ukf.batch_filter(measurements, saver = s)
        else:
            mu, cov = ukf.batch_filter(measurements)
        
        # perform RTS smoother
        xs, Ps, Ks = ukf.rts_smoother(mu, cov)
        
        if saver_verb:
            return mu, cov, xs, Ps, s 
        else:
            return mu, cov, xs, Ps


    def kalman_filter_full(self, trajectories, a, b, measurement_sigma, order, cov_factor, q, subsample_factor = 1):
        """
        Kalman filter
        
        Parameters
        ----------
        trajectories : pandas DataFrame
            DataFrame containing trajectories
        a : float
            Alpha value
        b : float   
            Beta value
        measurement_sigma : float
            Measurement sigma
        order : int
            Order
        cov_factor : float  
            Covariance factor
        q : float
            Process noise
        subsample_factor : int
            Subsample factor
        """
        
        trajectories = trajectories.sort_values(by=['frame', 'particle'])
        trajectories = trajectories.loc[trajectories.frame.isin(trajectories.frame.unique()[::subsample_factor])]
        trajectories['frame'] = np.array(trajectories.frame/subsample_factor).astype(int)
        kalman_rts_trajectories = trajectories.copy()
        
        r0s = trajectories.loc[trajectories.frame == trajectories.frame.min(), ['x', 'y']].values
        r1s = trajectories.loc[trajectories.frame == trajectories.frame.min() + 1, ['x', 'y']].values
        r2s = trajectories.loc[trajectories.frame == trajectories.frame.min() + 2, ['x', 'y']].values
        
        v0s = (r1s - r0s) * int(self.fps/subsample_factor)
        v1s = (r2s - r1s) * int(self.fps/subsample_factor)
        a0s = (v1s - v0s) * int(self.fps/subsample_factor)
        
        positions = trajectories.loc[:, ['x', 'y']].values.reshape(len(trajectories.frame.unique()), len(trajectories.particle.unique()), 2)
        
        mu, cov, xs, Ps = zip(*parallel(joblib.delayed(self.unscented_kalman_filter)(positions[:, i], r0s[i], v0s[i], a0s[i], a, b, measurement_sigma, order, cov_factor, q, subsample_factor, False)
                                for i in tqdm(range(self.n_instances), desc="UKF + RTS application on droplet trajejectories")))
        mu = np.array(mu)
        cov = np.array(cov)
        xs = np.array(xs)
        Ps = np.array(Ps)
        
        if order == 2:
            for i in trajectories.particle.unique():                    
                kalman_rts_trajectories.loc[kalman_rts_trajectories.particle == i, ['x']] = xs[i, :, 0]
                kalman_rts_trajectories.loc[kalman_rts_trajectories.particle == i, ['x_var']] = Ps[i, :, 0, 0]
                kalman_rts_trajectories.loc[kalman_rts_trajectories.particle == i, ['y']] = xs[i, :, 2]
                kalman_rts_trajectories.loc[kalman_rts_trajectories.particle == i, ['y_var']] = Ps[i, :, 2, 2]
        elif order == 3:
            for i in trajectories.particle.unique():
                kalman_rts_trajectories.loc[kalman_rts_trajectories.particle == i, ['x']] = xs[i, :, 0]
                kalman_rts_trajectories.loc[kalman_rts_trajectories.particle == i, ['x_var']] = Ps[i, :, 0, 0]
                kalman_rts_trajectories.loc[kalman_rts_trajectories.particle == i, ['y']] = xs[i, :, 3]
                kalman_rts_trajectories.loc[kalman_rts_trajectories.particle == i, ['y_var']] = Ps[i, :, 3, 3]
        return kalman_rts_trajectories