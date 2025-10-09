import os
import string
import json
import joblib
import multiprocessing
n_jobs = int(multiprocessing.cpu_count()*0.8)
#print(f"Running parallel jobs with {n_jobs} cores")
parallel = joblib.Parallel(n_jobs=n_jobs, backend='loky', verbose=0)
from tqdm import tqdm
import cv2
import subprocess
import sys

import matplotlib.cm as cm
import numpy as np
from scipy.optimize import curve_fit
import trackpy as tp
import pandas as pd
from scipy.spatial import KDTree
from scipy.special import i0
import yupi.stats as ys
from yupi import Trajectory, WindowType, DiffMethod
import warnings
from numba import njit
from pydivsufsort import divsufsort

warnings.filterwarnings('ignore', category = RuntimeWarning, message = ".*invalid value encountered in scalar power.*")


##################################################################################################################
#                                         USER INTERFACE FUNCTIONS                                               #
##################################################################################################################

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
        
def print_recap_analysis(choices, params):
    """
    Print a recap of the user's choices at the end of the analysis.
    """
    print("   --------------------------------------- RECAP ---------------------------------------")
    print(f"         Video selection:                  {choices.get('trajectory_name', 'N/A')}")
    print(f"         Order analysis:                   {'Enabled' if choices.get('order') else 'Disabled'}")
    print(f"         Shape analysis:                   {'Enabled' if choices.get('shape') else 'Disabled'}")
    print(f"         TAMSD analysis:                   {'Enabled' if choices.get('tamsd') else 'Disabled'}")
    print(f"         Speed analysis:                   {'Enabled' if choices.get('speed') else 'Disabled'}")
    print(f"         Turning angles analysis:          {'Enabled' if choices.get('turning') else 'Disabled'}")
    print(f"         Velocity Autocovariance analysis: {'Enabled' if choices.get('vacf') else 'Disabled'}")
    print(f"         Dimer distribution analysis:      {'Enabled' if choices.get('dimer') else 'Disabled'}")    
    print(f"         Local EP analysis:                {'Enabled' if choices.get('local_ep') else 'Disabled'}")
    print(f"         Run or Import                     {'Run' if choices.get('run') else 'Import'}")
    print(f"         Save plots                        {'True' if choices.get('save') else 'False'}")
    print(f"         Show plots                        {'True' if choices.get('show') else 'False'}")
    print(f"         Animated plots                    {'True' if choices.get('animated') else 'False'}")
    print("\n")
    print(f"         Number of particles: {params['n_particles']}, of which {len(params['blue_particle_idx'])} are blue and {len(params['red_particle_idx'])} are red")
    print(f"         The trajectory has {params['n_frames']} frames at {params['fps']} fps --> {params['n_frames']/params['fps']:.2f} s")
    print(f"         Windowed analysis: windows of {params['window_length']} s and stride of {params['stride_length']} s --> {params['n_windows']} steps")
    print(f"         The evolution is divided in the following stages:")
    
    hours, res = divmod(params['frames_stages'] / params['fps'], 3600)
    minutes, seconds = divmod(res, 60)
    for i in range(len(params['steps_plot'])):
        print(f"            Stage {i + 1} starts at: {int(hours[i])}h {int(minutes[i])}m {int(seconds[i])}s")


    print("   -------------------------------------------------------------------------------------")


##################################################################################################################
#                                              INITIALIZATION FUNCTIONS                                          #
##################################################################################################################


def get_analysis_parameters(trajectory_name, config_path="./analysis_config.json"):
    """
    Get video parameters from the JSON configuration file.
    
    Parameters
    ----------
    trajectory_name : str
        Name of the video selection.
    config_path : str, optional
        Path to the JSON configuration file (default: "./analysis_config.json").
    
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
            analysis_config = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        raise RuntimeError(f"Error loading video configuration: {e}")

    config = analysis_config.get(trajectory_name)
    if not config:
        raise ValueError(f"Unknown video selection: {trajectory_name}")

    # Return dictionary
    if trajectory_name in ['1b_&_1r_1', '1b_&_1r_2', '1b_&_1r_3']:
        return {
            "system_name": config["system_name"],
            "initial_offset_b": config["initial_offset_b"],
            "initial_offset_r": config["initial_offset_r"],
            "n_particles": config["n_particles"],
            "xmin_b": config["xmin_b"], "ymin_b": config["ymin_b"], "xmax_b": config["xmax_b"], "ymax_b": config["ymax_b"],
            "xmin_r": config["xmin_r"], "ymin_r": config["ymin_r"], "xmax_r": config["xmax_r"], "ymax_r": config["ymax_r"],
            "resolution": config["resolution"],
            "petri_diameter": config["petri_diameter"],
            "subsample_factor": config["subsample_factor"],
            "video_fps": config["video_fps"],
            "pdf_base_path": config["pdf_base_path"],
            "video_source_path_blue": config["video_source_path_blue"],
            "video_source_path_red": config["video_source_path_red"],
            "trajectory_path": config["trajectory_path"],
            "stages_seconds": config["stages_seconds"],
            "crop_verb": config["crop_verb"],
        }
    else:
        return {
            "system_name": config["system_name"],
            "initial_offset": config["initial_offset"],
            "n_particles": config["n_particles"],
            "xmin": config["xmin"], "ymin": config["ymin"], "xmax": config["xmax"], "ymax": config["ymax"],
            "resolution": config["resolution"],
            "petri_diameter": config["petri_diameter"],
            "subsample_factor": config["subsample_factor"],
            "video_fps": config["video_fps"],
            "pdf_base_path": config["pdf_base_path"],
            "video_source_path": config["video_source_path"],
            "trajectory_path": config["trajectory_path"],
            "stages_seconds": config["stages_seconds"],
            "crop_verb": config["crop_verb"]
        }


def ensure_video_and_trajectory_exist(video_selection, analysis_config_path='analysis_config.json', links_path='links.json'):
    with open(analysis_config_path) as f:
        analysis_config = json.load(f)
            
    video_path = analysis_config[video_selection]['video_source_path']
    if not os.path.isfile(video_path):
        print(f"Video file '{video_path}' not found. Attempting to download...")

        # Load links config
        with open('links.json', 'r') as f:
            links_config = json.load(f)

        # Get the download link for the video
        video_url = links_config[video_selection]['video_url']
        if not video_url:
            print(f"No download URL found for '{video_selection}' in links.json.")
            sys.exit(1)
        
        # Download using gdown
        try:
            subprocess.run(['gdown', '--fuzzy', video_url], check=True, cwd = os.path.dirname(video_path))
            if os.path.isfile(video_path):
                print(f"Download successful: {video_path}")
            else:
                print(f"Download attempted but '{video_path}' still not found.")
                sys.exit(1)
            
        except subprocess.CalledProcessError:
            print("gdown failed to download the file.")
            sys.exit(1)
    else:
        print(f"Video file '{video_path}' found.")
    
    trajectory_path = analysis_config[video_selection]['trajectory_path']
    if not os.path.isfile(trajectory_path):
        print(f"Trajectory file '{trajectory_path}' not found. Attempting to download...")

        # Load links config
        with open('links.json', 'r') as f:
            links_config = json.load(f)

        # Get the download link for the video
        trajectory_url = links_config[video_selection]['trajectory_url']
        if not trajectory_url:
            print(f"No download URL found for '{video_selection}' in links.json.")
            sys.exit(1)
        
        # Download using gdown
        try:
            subprocess.run(['gdown', '--fuzzy', trajectory_url], check=True, cwd = os.path.dirname(trajectory_path))
            if os.path.isfile(trajectory_path):
                print(f"Download successful: {trajectory_path}")
            else:
                print(f"Download attempted but '{trajectory_path}' still not found.")
                sys.exit(1)
            
        except subprocess.CalledProcessError:
            print("gdown failed to download the file.")
            sys.exit(1)
    else:
        print(f"Trajectory file '{trajectory_path}' found.")

def get_video_properties(trajectory_name, video_source_path_blue=None, video_source_path_red=None, video_source_path=None):
    """Retrieves video properties such as resolution, frame count, and FPS."""
    video_data = {}
    if trajectory_name in ['1b_&_1r_1', '1b_&_1r_2', '1b_&_1r_3']:
        video_blue = cv2.VideoCapture(video_source_path_blue, cv2.CAP_FFMPEG)
        video_blue.set(cv2.CAP_PROP_POS_FRAMES, 0)
        video_data['w_blue'] = int(video_blue.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_data['h_blue'] = int(video_blue.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_data['video_fps_blue'] = video_blue.get(cv2.CAP_PROP_FPS)
        video_data['n_frames_video_blue'] = int(video_blue.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f'Blue video has {video_data["n_frames_video_blue"]} frames with a resolution of {video_data["w_blue"]}x{video_data["h_blue"]} and a framerate of {video_data["video_fps_blue"]} fps')
                
        video_red = cv2.VideoCapture(video_source_path_red, cv2.CAP_FFMPEG)
        video_red.set(cv2.CAP_PROP_POS_FRAMES, 0)
        video_data['w_red'] = int(video_red.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_data['h_red'] = int(video_red.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_data['video_fps_red'] = video_red.get(cv2.CAP_PROP_FPS)
        video_data['n_frames_video_red'] = int(video_red.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f'Red video has {video_data["n_frames_video_red"]} frames with a resolution of {video_data["w_red"]}x{video_data["h_red"]} and a framerate of {video_data["video_fps_red"]} fps')
        return video_blue, video_red, video_data
    else:
        video = cv2.VideoCapture(video_source_path, cv2.CAP_FFMPEG)
        video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        video_data['w'] = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_data['h'] = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_data['video_fps'] = video.get(cv2.CAP_PROP_FPS)
        video_data['n_frames_video'] = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f'Video has {video_data["n_frames_video"]} frames with a resolution of {video_data["w"]}x{video_data["h"]} and a framerate of {video_data["video_fps"]} fps')    
        return video, video_data

def create_masks(n_particles, red_particle_idx):
    """Creates a boolean mask for red particles and assigns colors."""
    red_mask = np.zeros(n_particles, dtype=bool)
    red_mask[red_particle_idx] = True
    colors = np.array(['b'] * n_particles)
    colors[red_particle_idx] = 'r'
    return red_mask, colors

def compute_kinematics(trajectories, fps, n_frames, n_particles):
    """Computes droplet velocities and accelerations."""
    velocities = np.zeros((n_frames, n_particles, 2))
    accelerations = np.zeros_like(velocities)

    for i in tqdm(trajectories.particle.unique(), desc = '   Computing velocities and accelerations'):
        p = trajectories.loc[trajectories.particle == i]
        temp = Trajectory(p.x, p.y, dt = 1/fps, traj_id = i,
                          diff_est = {'method': DiffMethod.LINEAR_DIFF,
                                      'window_type': WindowType.FORWARD})
        velocities[:, i] = temp.v
        accelerations[:, i] = temp.a

    # Invert y-component of velocities if necessary
    velocities[:, :, 1] *= -1
    return velocities, accelerations

def compute_properties(trajectories, n_frames, n_particles, velocities):
    """Computes droplet orientations, radii, and eccentricities."""
    orientations = velocities / np.linalg.norm(velocities, axis=2).reshape(n_frames, n_particles, 1)
    radii = trajectories.r.values.reshape(n_frames, n_particles)
    eccentricity = trajectories.eccentricity.values.reshape(n_frames, n_particles)
    return orientations, radii, eccentricity

def create_directories(params):
    """Creates necessary directories for results and analysis."""
    params['res_path'] = f"./analysis_results/{params['trajectory_name']}/results_wind_{params['window_length']}_subsample_{params['subsample_factor']}"
    params['analysis_data_path'] = f"./analysis_results/all_the_others/{params['trajectory_name']}/analysis_data_wind_{params['window_length']}_subsample_{params['subsample_factor']}"
    params['pdf_res_path'] = f"{params['pdf_base_path']}/images/{params['trajectory_name']}/results_wind_{params['window_length']}_subsample_{params['subsample_factor']}"
    
    os.makedirs(params['res_path'], exist_ok = True)
    os.makedirs(params['analysis_data_path'], exist_ok = True)
    os.makedirs(params['pdf_res_path'] , exist_ok = True)

    for path in params['folder_names']:
        os.makedirs(params['res_path'] + '/' + path, exist_ok = True)
        os.makedirs(params['analysis_data_path'] + '/' + path, exist_ok = True)
        os.makedirs(params['pdf_res_path'] + '/' + path, exist_ok = True)
    
    return params

def compute_windowed_analysis(frames, fps, window_length, stride_length, frames_stages):
    """Computes parameters for windowed analysis."""
    n_frames_window = int(window_length * fps)
    n_frames_stride = int(stride_length * fps)
    startFrames = np.arange(frames[0], frames[-1] - n_frames_window, n_frames_stride, dtype=int)
    window_center_sec = (startFrames + n_frames_window / 2) / fps
    endFrames = startFrames + n_frames_window
    n_windows = len(startFrames)
    n_stages = len(frames_stages)
    steps_plot = np.array([find_nearest(startFrames + n_frames_window / 2, frame) for frame in frames_stages])
    
    return startFrames, window_center_sec, endFrames, n_windows, n_stages, steps_plot

def generate_plot_styles(n_stages):
    """Generates color styles and labels for plotting."""
    alphas = np.linspace(0.2, 0.8, n_stages)
    shades_of_blue = [cm.colors.rgb2hex(cm.Blues(0.2 + 0.8 * i / (n_stages - 1))) for i in range(n_stages)]
    shades_of_red = [cm.colors.rgb2hex(cm.Reds(0.2 + 0.8 * i / (n_stages - 1))) for i in range(n_stages)]
    
    default_kwargs_blue = [{'color': shades_of_blue[i], 'density': True} for i in range(n_stages)]
    default_kwargs_red = [{'color': shades_of_red[i], 'density': True} for i in range(n_stages)]
    
    letter_labels = [f'{letter})' for letter in string.ascii_lowercase]
    stages_shades = ['#8ecae6', '#219ebc', '#023047', '#fb8500', '#ffb703']
    
    return shades_of_blue, default_kwargs_blue, shades_of_red, default_kwargs_red, letter_labels, stages_shades


##################################################################################################################
#                                               MISC FUNCTIONS                                                   #
##################################################################################################################

def onClick(event):
    global anim_running
    if anim_running:
        ani.event_source.stop()
        anim_running = False
    else:
        ani.event_source.start()
        anim_running = True

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def trim_up_to_char(s, char):
    index = s.find(char)
    if index != -1:
        return s[:index]
    return s

def get_frame(cap, frame, x1, y1, x2, y2, resolution, crop_verb):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ret, image = cap.read()

    if not ret or image is None:
        print(f"Warning: Could not read frame {frame}. Returning black frame.")
        return np.zeros((resolution, resolution, 3), dtype=np.uint8)

    try:
        npImage = np.array(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    except cv2.error as e:
        print(f"Warning: cvtColor failed on frame {frame}: {e}")
        return np.zeros((resolution, resolution, 3), dtype=np.uint8)

    if crop_verb:
        npImage = npImage[y1:y2, x1:x2]

    try:
        npImage = cv2.resize(npImage, (resolution, resolution))
    except cv2.error as e:
        print(f"Warning: resize failed on frame {frame}: {e}")
        return np.zeros((resolution, resolution, 3), dtype=np.uint8)

    return npImage


def get_frame_increased_contrast(cap, frame, x1, y1, x2, y2, w, h, resolution):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ret, image = cap.read()
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit = 3, tileGridSize = (2, 2))
    cl = clahe.apply(l_channel)
    image = cv2.merge((cl,a,b))
    image = cv2.cvtColor(image, cv2.COLOR_LAB2RGB)
    image = cv2.filter2D(image, -1, np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])) 
    image = cv2.resize(image, (resolution, resolution))
    image = image[y1 + 50:y2 - 50, x1 + 50:x2 - 50]
    image = cv2.resize(image, (resolution, resolution))
    return image

def get_frame_rgb_circular_crop(video, frame, xmin, ymin, xmax, ymax, resolution, crop_verb):
    video.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ret, image = video.read()

    if not ret or image is None:
        print(f"Warning: Could not read frame {frame}. Returning white frame.")
        return np.full((ymax - ymin, xmax - xmin, 3), 255, dtype=np.uint8)

    try:
        image = np.array(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    except cv2.error as e:
        print(f"Warning: cvtColor failed on frame {frame}: {e}")
        return np.full((ymax - ymin, xmax - xmin, 3), 255, dtype=np.uint8)
    
    if crop_verb:
        image = image[ymin:ymax, xmin:xmax]
        
    try:
        image = cv2.resize(image, (resolution, resolution))
    except cv2.error as e:
        print(f"Warning: resize failed on frame {frame}: {e}")
        return np.full((ymax - ymin, xmax - xmin, 3), 255, dtype=np.uint8)

    try:
        if crop_verb:
            side_length = resolution
            center = (resolution // 2, resolution // 2)
            radius = side_length // 2
            y, x = np.ogrid[:resolution, :resolution]
            mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2
            circular_image = np.full_like(image, 255)  # White background
            circular_image[mask] = image[mask]
        else:
            side_length = xmax - xmin
            center = (xmin + side_length // 2, ymin + side_length // 2)
            radius = side_length // 2
            y, x = np.ogrid[:image.shape[0], :image.shape[1]]
            mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2
            circular_image = np.full_like(image, 255)  # White background
            circular_image[mask] = image[mask]
            circular_image = circular_image[ymin:ymax, xmin:xmax]
    except Exception as e:
        print(f"Warning: cropping or masking failed on frame {frame}: {e}")
        return np.full((ymax - ymin, xmax - xmin, 3), 255, dtype=np.uint8)

    return circular_image

# 2D Maxwell-Boltzmann distribution
def MB_2D(v, sigma):
    return v/(sigma**2) * np.exp(-v**2/(2*sigma**2))

# Generalized 2D Maxwell-Boltzmann distribution
def MB_2D_generalized(v, sigma, beta, A):
    return A * v * np.exp(-v**beta/(2*sigma**beta))

# Normal distribution
def normal_distr(x, sigma, mu):
    return 1/(np.sqrt(2*np.pi)*sigma) * np.exp(-0.5*((x-mu)/sigma)**2)

# Wrapped lorentzian distribution
def wrapped_lorentzian_distr(theta, gamma, mu):
    return 1/(2*np.pi) * np.sinh(gamma) / (np.cosh(gamma) - np.cos(theta - mu))

# Lorentzian distribution
def lorentzian_distr(x, gamma, x0):
    return 1/np.pi * gamma / ((x-x0)**2 + gamma**2)

def von_mises(x, kappa, mu):
    return np.exp(kappa * np.cos(x - mu)) / (2 * np.pi * i0(kappa))

# Power Law distribution
def powerLaw(x, a, k):
    return a*x**k

# Exponential distribution
def exp(t, A, tau):
    return A * np.exp(-t/tau)

# Histogram fit
def fit_hist(y, bin_centers, distribution, p0_, maxfev_):
    ret, pcov = curve_fit(distribution, bin_centers, y, p0 = p0_, maxfev = maxfev_)
    ret_std = np.sqrt(np.diag(pcov))
    fit_results = np.array([ret, ret_std]).T
    y_fit = distribution(bin_centers, *fit_results[:, 0])
    r2 = 1 - (np.sum((y - y_fit) ** 2) / np.sum((y - np.mean(y)) ** 2))
    return fit_results, r2

def get_trajs(trajs, fps, pxDimension):
    yupi_trajs = []
    for i in trajs.particle.unique():
        p = trajs.loc[trajs.particle == i, ['x','y']]
        temp_traj = Trajectory(p.x*pxDimension, p.y*pxDimension, 
                               dt = 1/fps, traj_id=i, 
                               diff_est={'method':DiffMethod.LINEAR_DIFF, 
                               'window_type': WindowType.CENTRAL})
        
        yupi_trajs.append(temp_traj)
    return yupi_trajs

def powerLawFit(f, x, N, yerr, maxfev_):
    if N == 1:
        x = np.array(x)
        f = np.array(f).reshape(-1)
        ret = np.zeros((2, 2))
        if yerr is None:
            ret[0], pcov = curve_fit(powerLaw, x, f, p0 = [1., 1.], maxfev = maxfev_)
        else:
            ret[0], pcov = curve_fit(powerLaw, x, f, p0 = [1., 1.], sigma = yerr, maxfev = maxfev_)
        ret[1] = np.sqrt(np.diag(pcov))
    else:
        ret = np.zeros((N, 2, 2))
        for i in range(N):
            if yerr is None:
                ret[i, 0], pcov = curve_fit(powerLaw, x, f[i], p0 = [1., 1.], maxfev = maxfev_)
            else:
                ret[i, 0], pcov = curve_fit(powerLaw, x, f[i], p0 = [1., 1.], sigma = yerr, maxfev = maxfev_)
            ret[i, 1] = np.sqrt(np.diag(pcov))
    return ret


##################################################################################################################
#                                              ANALYSES FUNCTIONS                                                #
##################################################################################################################

def get_neighbours_props(pos, r_mean):
    n_particles = pos.shape[0]
    ktree = KDTree(pos)
    n_of_neighbors = np.zeros(n_particles)
    theta_ij = []
    for i in range(n_particles):
        neighbors = ktree.query_ball_point(pos[i], r = 2*r_mean*1.5, return_sorted = False, return_length = False)
        neighbors = [n for n in neighbors if n != i]
        #print(i, neighbors)
        n_of_neighbors[i] = len(neighbors)
        if len(neighbors) >= 1:
            for k, j in enumerate(neighbors):
                r_ij = pos[j] - pos[i]
                temp_theta = np.arctan2(r_ij[1], r_ij[0])
                if k == 0:
                    theta_0 = temp_theta
                temp_theta -= theta_0
                theta_ij.append(temp_theta)
    return n_of_neighbors, theta_ij

# uses the first neighbor as reference angle --> theta_0 = 0 --> hex_order = 1/6 
def compute_hex_order_frame(pos, r_mean):
    n_particles = pos.shape[0]
    n_of_neighbors = np.zeros(n_particles)
    ktree = KDTree(pos)
    hex_order = np.zeros(n_particles, dtype = complex)
    for i in range(n_particles):
        neighbors = ktree.query_ball_point(pos[i], r = 2*r_mean*1.5, return_sorted = False, return_length = False)
        neighbors = [n for n in neighbors if n != i]
        n_of_neighbors[i] = len(neighbors)
        if len(neighbors) >= 1:
            for k, j in enumerate(neighbors):
                r_ij = pos[j] - pos[i]
                theta_ij = np.arctan2(r_ij[1], r_ij[0])
                if k == 0:
                    theta_0 = theta_ij
                theta_ij -= theta_0
                hex_order[i] += np.exp(6j*theta_ij)/6
    return np.mean(np.real(hex_order)), np.mean(np.imag(hex_order)), np.mean(n_of_neighbors), np.std(n_of_neighbors)

def compute_hex_order(positions, r_mean, description):
    res = parallel(joblib.delayed(compute_hex_order_frame)(positions[frame], r_mean[frame])
                      for frame in tqdm(range(positions.shape[0]), desc = description))
    return np.array(res)

def get_imsd(trajs, pxDimension, fps, maxLagtime, fit_range, id_start_fit, id_end_fit):
    imsd = tp.imsd(trajs, mpp = pxDimension, fps = fps, max_lagtime = maxLagtime)
    # fit the IMSD in the fit_range
    imsd_to_fit = imsd.iloc[id_start_fit:id_end_fit]
    pw_exp = powerLawFit(imsd_to_fit, fit_range, len(trajs.particle.unique()), None, maxfev_ = 10000)
    return imsd, pw_exp

def get_emsd(imsd, fit_range, id_start_fit, id_end_fit):
    # compute the EMSD
    MSD = np.array(imsd)
    MSD = [MSD.mean(axis = 1), MSD.std(axis = 1)]
    # fit the EMSD in the fit_range
    pw_exp = powerLawFit(MSD[0][id_start_fit:id_end_fit], fit_range, 1, MSD[1][id_start_fit:id_end_fit], maxfev_ = 10000)
    return MSD, pw_exp

def get_imsd_windowed(nSteps, startFrames, endFrames, trajs, pxDimension, fps, maxLagtime, fit_range, id_start_fit, id_end_fit, progress_verb):
    if progress_verb:
        MSD_wind, pw_exp_wind = zip(*parallel(joblib.delayed(get_imsd)(trajs.iloc[(trajs.index >= startFrames[k]) & (trajs.index < endFrames[k])], pxDimension, fps, maxLagtime, fit_range, id_start_fit, id_end_fit) for k in tqdm(range(nSteps), desc = "Computing windowed MSD")))
    else:
        MSD_wind, pw_exp_wind = zip(*parallel(joblib.delayed(get_imsd)(trajs.iloc[(trajs.index >= startFrames[k]) & (trajs.index < endFrames[k])], pxDimension, fps, maxLagtime, fit_range, id_start_fit, id_end_fit) for k in range(nSteps)))
    return MSD_wind, np.array(pw_exp_wind)


def get_emsd_windowed(imsd, x, fps, red_mask, nSteps, maxLagtime, fit_range, id_start_fit, id_end_fit, progress_verb):
    EMSD_wind = np.array(imsd)
    EMSD_wind_b = [EMSD_wind[:, :, ~red_mask].mean(axis = 2), EMSD_wind[:, :, ~red_mask].std(axis = 2)]
    EMSD_wind_r = [EMSD_wind[:, :, red_mask].mean(axis = 2), EMSD_wind[:, :, red_mask].std(axis = 2)]
    del EMSD_wind

    # diffusive region of the MSD
    pw_exp_wind_b = np.zeros((nSteps, 2, 2))
    pw_exp_wind_r = np.zeros((nSteps, 2, 2))
    if progress_verb:
        for i in tqdm(range(nSteps), desc = "Fitting with power law"):
            pw_exp_wind_b[i] = powerLawFit(EMSD_wind_b[0][i, id_start_fit:id_end_fit], x, 1, EMSD_wind_b[1][i, id_start_fit:id_end_fit], maxfev_ = 10000)
            pw_exp_wind_r[i] = powerLawFit(EMSD_wind_r[0][i, id_start_fit:id_end_fit], x, 1, EMSD_wind_r[1][i, id_start_fit:id_end_fit], maxfev_ = 10000)
    else:
        for i in range(nSteps):
            pw_exp_wind_b[i] = powerLawFit(EMSD_wind_b[0][i, id_start_fit:id_end_fit], x, 1, EMSD_wind_b[1][i, id_start_fit:id_end_fit], maxfev_ = 10000)
            pw_exp_wind_r[i] = powerLawFit(EMSD_wind_r[0][i, id_start_fit:id_end_fit], x, 1, EMSD_wind_r[1][i, id_start_fit:id_end_fit], maxfev_ = 10000)
    return EMSD_wind_b, EMSD_wind_r, pw_exp_wind_b, pw_exp_wind_r


def get_emsd_wind(trajs, pxDimension, fps, maxLagtime, x_ballistic, x_diffusive, params):
    imsd = tp.imsd(trajs, mpp = pxDimension, fps = fps, max_lagtime = maxLagtime)
    MSD = np.array(imsd)
    MSD = [MSD.mean(axis = 1), MSD.std(axis = 1)]
    # fit the EMSD in the ballistic_range and diffusive_range
    
    id_start_fit_ballistic = int(x_ballistic[0]*params['fps']) - 1
    id_end_fit_ballistic = int(x_ballistic[-1]*params['fps'])
    
    id_start_fit_diffusive = int(x_diffusive[0]*params['fps']) - 1
    id_end_fit_diffusive = int(x_diffusive[-1]*params['fps'])
    
    pw_exp_ballistic = powerLawFit(MSD[0][id_start_fit_ballistic:id_end_fit_ballistic], x_ballistic, 1, MSD[1][id_start_fit_ballistic:id_end_fit_ballistic], maxfev_ = 10000)
    pw_exp_diffusive = powerLawFit(MSD[0][id_start_fit_diffusive:id_end_fit_diffusive], x_diffusive, 1, MSD[1][id_start_fit_diffusive:id_end_fit_diffusive], maxfev_ = 10000)
    return MSD, pw_exp_ballistic, pw_exp_diffusive


def get_emsd_windowed_v2(nSteps, startFrames, endFrames, trajs, pxDimension, fps, maxLagtime, x_ballistic, x_diffusive, params, progress_verb, description):
    if progress_verb:
        MSD_wind, pw_exp_wind_ballistic, pw_exp_wind_diffusive = zip(*parallel(
            joblib.delayed(get_emsd_wind)(
                trajs.loc[startFrames[k]:endFrames[k]-1],
                pxDimension, fps, maxLagtime, x_ballistic, x_diffusive, params
            ) for k in tqdm(range(nSteps), desc = description)
        ))
    else:
        MSD_wind, pw_exp_wind_ballistic, pw_exp_wind_diffusive = zip(*parallel(
            joblib.delayed(get_emsd_wind)(
                trajs.loc[startFrames[k]:endFrames[k]-1],
                pxDimension, fps, maxLagtime, x_ballistic, x_diffusive, params
            ) for k in range(nSteps)
        ))
    return np.array(MSD_wind), np.array(pw_exp_wind_ballistic), np.array(pw_exp_wind_diffusive)

# get speed distributions windowed in time
def speed_wind(trajs_wind, fps, pxDimension, speed_bins, speed_bin_centers):
    speed = ys.speed_ensemble(get_trajs(trajs_wind, fps, pxDimension), step = 1)
    mean_speed = np.mean(speed)
    std_speed = np.std(speed)
    speed_distr = np.histogram(speed, bins = speed_bins, density = True)[0]

    # fit windowed speed distribution with 2D MB and Generalized 2D MB distributions
    fit_results_wind, r2_wind = fit_hist(speed_distr, speed_bin_centers, MB_2D, [1.], maxfev_ = 10000)
    fit_results_wind_g, r2_g_wind = fit_hist(speed_distr, speed_bin_centers, MB_2D_generalized, [1., 2., 1.], maxfev_ = 10000)
    return mean_speed, std_speed, speed_distr, fit_results_wind, r2_wind, fit_results_wind_g, r2_g_wind

def speed_windowed(nSteps, startFrames, endFrames, trajs, fps, pxDimension, speed_bins, speed_bin_centers, progress_verb, description):
    if progress_verb:    
        mean_speed, std_speed, speed_distr, fit_results_wind, r2_wind, fit_results_wind_g, r2_g_wind = zip(*parallel(
            joblib.delayed(speed_wind)(
                trajs.loc[startFrames[k]:endFrames[k]-1],
                fps, pxDimension, speed_bins, speed_bin_centers
                ) for k in tqdm(range(nSteps), desc = description)
        ))
    else:
        mean_speed, std_speed, speed_distr, fit_results_wind, r2_wind, fit_results_wind_g, r2_g_wind = zip(*parallel(
            joblib.delayed(speed_wind)(
                trajs.loc[startFrames[k]:endFrames[k]-1],
                fps, pxDimension, speed_bins, speed_bin_centers
                ) for k in range(nSteps)
        ))
    mean_speed = np.array(mean_speed)
    std_speed = np.array(std_speed)
    speed_distr = np.array(speed_distr)
    fit_results_wind = np.array(fit_results_wind)
    r2_wind = np.array(r2_wind)
    fit_results_wind_g = np.array(fit_results_wind_g)
    r2_g_wind = np.array(r2_g_wind)
    
    return mean_speed, std_speed, speed_distr, fit_results_wind, r2_wind, fit_results_wind_g, r2_g_wind

def turn_angl_wind(trajs_wind, fps, pxDimension, turn_angles_bins, turn_angles_bin_centers):
    yupi_trajs = get_trajs(trajs_wind, fps, pxDimension)
    theta = ys.turning_angles_ensemble(yupi_trajs, centered = True)
    turn_angles = np.histogram(theta, bins = turn_angles_bins, density = True)[0]
    gaussian_fit_results_wind, gaussian_r2_wind = fit_hist(turn_angles, turn_angles_bin_centers, von_mises, [1., 0.], maxfev_ = 10000)
    lorentzian_fit_results_wind, lorentzian_r2_wind = fit_hist(turn_angles, turn_angles_bin_centers, wrapped_lorentzian_distr, [1., 0.], maxfev_ = 10000)
    return turn_angles, gaussian_fit_results_wind, gaussian_r2_wind, lorentzian_fit_results_wind, lorentzian_r2_wind 

def turning_angles_windowed(nSteps, startFrames, endFrames, trajs, fps, pxDimension, turn_angles_bins, turn_angles_bin_centers, progress_verb, description):
    if progress_verb: 
        turn_angles, gaussian_fit_results_wind, gaussian_r2_wind, lorentzian_fit_results_wind, lorentzian_r2_wind = zip(*parallel(
            joblib.delayed(turn_angl_wind)(trajs.loc[startFrames[k]:endFrames[k]-1],
                fps, pxDimension, turn_angles_bins, turn_angles_bin_centers
                ) for k in tqdm(range(nSteps), desc = description)
        ))
    else:
        turn_angles, gaussian_fit_results_wind, gaussian_r2_wind, lorentzian_fit_results_wind, lorentzian_r2_wind = zip(*parallel(
            joblib.delayed(turn_angl_wind)(trajs.loc[startFrames[k]:endFrames[k]-1],
                fps, pxDimension, turn_angles_bins, turn_angles_bin_centers
                ) for k in range(nSteps)
        ))
        
    turn_angles = np.array(turn_angles)
    gaussian_fit_results_wind = np.array(gaussian_fit_results_wind)
    gaussian_r2_wind = np.array(gaussian_r2_wind)
    lorentzian_fit_results_wind = np.array(lorentzian_fit_results_wind)
    lorentzian_r2_wind = np.array(lorentzian_r2_wind)
        
    return turn_angles, gaussian_fit_results_wind, gaussian_r2_wind, lorentzian_fit_results_wind, lorentzian_r2_wind

def vacf_yupi_modified(trajs_wind, fps, pxDimension, lag, vacf_time_verb = False):
    yupi_trajs = get_trajs(trajs_wind, fps, pxDimension)
    _vacf = []
    for traj in yupi_trajs:
        # Cartesian velocity components
        v = traj.v

        # Compute vacf for a single trajectory
        current_vacf = np.empty(lag)
        for lag_ in range(lag):
            # Multiply components given lag
            if lag_ == 0:
                v1, v2 = v, v
            else:
                v1, v2 = v[:-lag_], v[lag_:]
            v1v2 = (v1 - v1.mean(axis=0)) * (v2 - v2.mean(axis=0))

            # Dot product for a given lag time
            v1_dot_v2 = np.sum(v1v2, axis=1)

            # Averaging over a single realization
            current_vacf[lag_] = np.mean(v1_dot_v2)

        # Append the vacf for a every single realization
        _vacf.append(current_vacf)
    # Aranspose to have time/trials as first/second axis
    _vacf = np.transpose(_vacf)
    if vacf_time_verb:
        return _vacf
    else:
        vacf_mean = np.mean(_vacf, axis=1)  # Mean
        vacf_std = np.std(_vacf, axis=1)  # Standard deviation
        return vacf_mean, vacf_std
    
def vacf_windowed(trajs, nSteps, startFrames, endFrames, fps, pxDimension, maxLagtime, progress_verb, description):
    if progress_verb:
        vacf_wind, vacf_std_wind = zip(*parallel(
            joblib.delayed(vacf_yupi_modified)(trajs.loc[startFrames[k]:endFrames[k] - 1],
                            fps, pxDimension, maxLagtime) 
            for k in tqdm(range(nSteps), desc = description)
        ))
    else:
        vacf_wind, vacf_std_wind = zip(*parallel(
            joblib.delayed(vacf_yupi_modified)(trajs.loc[startFrames[k]:endFrames[k] - 1],
                            fps, pxDimension, maxLagtime) 
            for k in range(nSteps)
        ))

    vacf_wind = np.array(vacf_wind)
    vacf_std_wind = np.array(vacf_std_wind)
    
    return vacf_wind, vacf_std_wind


def precompute_neighbour_ids(coords_orig, coords_target, same_set):
    """Precompute nearest neighbour IDs for all frames."""
    ids_neighbours_list = np.zeros((len(coords_target), len(coords_orig[0])), dtype=np.int32)
    if same_set:
        for i in range(len(coords_target)):
            kd = KDTree(coords_target[i])
            # [1] to get the ids, [:, 1] to get the first nearest neighbour
            ids_neighbours_list[i] = kd.query(coords_orig[i], k = 2)[1][:, 1]

    else:
        for i in range(len(coords_target)):
            kd = KDTree(coords_target[i])
            # [1] to get the ids, [:, 0] to get the first nearest neighbour
            ids_neighbours_list[i] = kd.query(coords_orig[i], k = 2)[1][:, 0]
   
    return ids_neighbours_list

def dimer_distr_frame(coords_orig, coords_target, id_nearest_neighbour, pxDimension, r_bins, same_set):
    """Compute relative positions and generate histogram."""
    
    angles = -np.arctan2(coords_target[id_nearest_neighbour, 1] - coords_orig[:, 1], coords_target[id_nearest_neighbour, 0] - coords_orig[:, 0])
    
    # Translate coordinates
    relative_positions = coords_target[:, np.newaxis, :] - coords_orig[np.newaxis, :, :]
    
    # Mask diagonal (droplet relative to itself)
    if same_set:
        np.fill_diagonal(relative_positions[..., 0], np.nan)
        np.fill_diagonal(relative_positions[..., 1], np.nan)
        
    # Rotate coordinates
    x_new = relative_positions[..., 0] * np.cos(angles) - relative_positions[..., 1] * np.sin(angles)
    y_new = relative_positions[..., 0] * np.sin(angles) + relative_positions[..., 1] * np.cos(angles)
    relative_positions = np.stack([x_new, y_new], axis = 2).reshape(-1, 2) * pxDimension
    
    # Return 2D histogram transposed
    return np.histogram2d(relative_positions[:, 0], relative_positions[:, 1], bins = [r_bins, r_bins], density=True)[0].T

def dimer_distr_batch(ids_neighbours_list, coords_orig, coords_target, pxDimension, r_bins, same_set):
    """Compute dimer distributions for a batch of frames."""
    res = np.zeros((len(coords_orig), len(r_bins) - 1, len(r_bins) - 1), dtype=np.float16)
    for frame in range(len(coords_orig)):	
        res[frame] = dimer_distr_frame(coords_orig[frame], coords_target[frame], ids_neighbours_list[frame], pxDimension, r_bins, same_set)
    return np.mean(res, axis = 0)

def compute_windowed_dimer_distribution(coords_orig, coords_target, r_bins, pxDimension, same_set, n_windows, startFrames, endFrames, description):
    """Compute dimer distributions for windowed time frames."""
    
    # Precompute nearest neighbour IDs
    ids_neighbours_list = precompute_neighbour_ids(coords_orig, coords_target, same_set)
    
    # Compute dimer distributions for each n_frames_window
    dimer_distr_windowed = parallel(joblib.delayed(dimer_distr_batch)(
                                    ids_neighbours_list[startFrames[i]:endFrames[i]],\
                                      coords_orig[startFrames[i]:endFrames[i]],\
                                      coords_target[startFrames[i]:endFrames[i]],\
                                      pxDimension, r_bins, same_set)
                                for i in tqdm(range(n_windows), desc = description))
    
    return np.array(dimer_distr_windowed)


def get_grid_values_frame(grid_centers, points, d, dim):
    points = np.atleast_2d(points)
    points_expanded = points[np.newaxis, :, :]
    within_x = (grid_centers[:, :, 0] - d <= points_expanded[:, :, 0]) & (points_expanded[:, :, 0] <= grid_centers[:, :, 0] + d)
    within_y = (grid_centers[:, :, 1] - d <= points_expanded[:, :, 1]) & (points_expanded[:, :, 1] <= grid_centers[:, :, 1] + d)
    within_area = within_x & within_y
	
    grid_values_frame = np.any(within_area, axis=1).astype(np.int8).reshape(dim, dim)
    return grid_values_frame

def get_local_trajectories(frames, grid_centers, droplet_positions, d, dim):
    res = parallel(joblib.delayed(get_grid_values_frame)(grid_centers, droplet_positions[frame], d, dim) for frame in tqdm(frames))
    return np.array(res).reshape(len(frames), dim, dim)

def arr_to_str(arr):
	return str(list(arr)).replace(', ', '').replace('[', '').replace(']', '')

def time_shuffled_segments_bootstrapping(trajectory, segment_length):
    segments = [trajectory[i*segment_length:(i+1)*segment_length] for i in range(len(trajectory) // segment_length)]
    np.random.shuffle(segments)
    return np.concatenate(segments)

@njit
def lexicographical_compare_less(arr1, arr2):
	'''Optimized lexicographical comparison using Numba.'''
	n = min(len(arr1), len(arr2))
	for i in range(n):
		if arr1[i] < arr2[i]:
			return True
		elif arr1[i] > arr2[i]:
			return False
	return len(arr1) < len(arr2)

@njit
def array_equal_numba(arr1, arr2):
	'''Optimized array equality check using Numba.'''
	if arr1.shape != arr2.shape:
		return False
	for i in range(arr1.size):
		if arr1.flat[i] != arr2.flat[i]:
			return False
	return True

@njit
def binary_search_suffix(prefix, B, suffix_array):
	'''Binary search for the longest prefix match in B using its suffix array.'''
	low, high = 0, len(suffix_array) - 1
	prefix_len = len(prefix)

	while low <= high:
		mid = (low + high) // 2
		suffix_start = suffix_array[mid]
		suffix = B[suffix_start:suffix_start + prefix_len]

		if array_equal_numba(suffix, prefix):
			return True
		elif lexicographical_compare_less(suffix, prefix):
			low = mid + 1
		else:
			high = mid - 1

	return False

@njit
def longest_prefix_in_B(A, B, suffix_array):
	'''Finds the longest prefix of A that appears in B using suffix array and binary search.'''
	max_len = 0
	len_A = len(A)

	for i in range(1, len_A + 1):
		prefix = A[:i]

		# Use the Numba optimized binary search to check if the prefix is in B
		if binary_search_suffix(prefix, B, suffix_array):
			max_len = i
		else:
			break

	return A[:max_len]

def cross_parsing_complexity(A, B):
	'''Parse sequence A by extracting longest substrings found in B.'''
	# Precompute suffix array and LCP array once
	suffix_array = divsufsort(B)
	
	parsed_output = []
	while len(A) > 0:
		substr = longest_prefix_in_B(A, B, suffix_array)
		if len(substr) > 0:
			parsed_output.append(substr)
			A = A[len(substr):]
		else:
			break
	return parsed_output, len(parsed_output)

def local_entropy_production_tile(local_traj, segment_length):
	bootstrapped_traj = time_shuffled_segments_bootstrapping(local_traj, segment_length)

	xi_i = local_traj
	xi_i_R = local_traj[::-1]
	xi_i_prime = bootstrapped_traj
	N = len(xi_i_prime)
	
	_, factors_R = cross_parsing_complexity(xi_i_prime, xi_i_R)
	_, factors = cross_parsing_complexity(xi_i_prime, xi_i)

	return N, (np.log(N)/N) * (factors_R - factors)

def local_entropy_production(local_trajectories, segment_length, show_progress):
	n_frames = local_trajectories.shape[0]
	local_trajectories_flat = local_trajectories.reshape(n_frames, local_trajectories.shape[1]**2)
	if show_progress:
		_, local_ep = zip(*parallel(
							joblib.delayed(local_entropy_production_tile)(local_trajectories_flat[:, i], segment_length)
				for i in tqdm(range(local_trajectories_flat.shape[1]))))
	else:
		_, local_ep = zip(*parallel(
							joblib.delayed(local_entropy_production_tile)(local_trajectories_flat[:, i], segment_length)
				for i in range(local_trajectories_flat.shape[1]))
		)
	return np.array(local_ep).reshape(local_trajectories.shape[1], local_trajectories.shape[2])



def local_entropy_prod_wind(local_trajectories, segment_length):
    n_frames = local_trajectories.shape[0]
    local_ep = np.zeros((local_trajectories.shape[1], local_trajectories.shape[2]))
    for i in range(local_trajectories.shape[1]):
        for j in range(local_trajectories.shape[2]):
            N, local_ep[i, j] = local_entropy_production_tile(local_trajectories[:, i, j], segment_length)
    return local_ep

def local_entropy_production_windowed(local_trajectories, segment_length, window, n_steps):
	local_ep = parallel(
		joblib.delayed(local_entropy_prod_wind)(local_trajectories[i*window:(i+1)*window], segment_length)
		for i in tqdm(range(n_steps))
	)
	return np.array(local_ep)

def check_convergence(local_traj, segment_length):
	N, local_entropy = zip(*parallel(
			joblib.delayed(local_entropy_production_tile)(local_traj[:i], segment_length)
			for i in tqdm(range(5000, len(local_traj), 1000))))
	return np.array(N), np.array(local_entropy)
