import os
import json
import string
import time

import matplotlib as mpl
mpl.rcParams['image.cmap'] = 'gray'
import matplotlib.pyplot as plt
from matplotlib.transforms import ScaledTranslation
import matplotlib.animation
plt.rcParams.update({
	'font.size': 15,               # General font size
	'axes.titlesize': 14,          # Title font size
	'axes.labelsize': 12,          # Axis label font size
	'legend.fontsize': 10,         # Legend font size
	'xtick.labelsize': 10,         # X-axis label size
	'ytick.labelsize': 10          # Y-axis label size
})       

import numpy as np
import pandas as pd
from tqdm import tqdm
tqdm.pandas()

from tracking_utils import ask_options, ask_yesno, user_message, print_recap_tracking, get_video_parameters, get_n_errors, draw_polygons, TrackingVideo, onClick

def main():
    print("\nWelcome to DropleX, the Python tool for tracking and analysis of active particles from videos! \nLet's start by selecting the video you want to track and the part of the tracking pipeline you want to run. \nPlease follow the instructions below. \n")

    # list of mp4 files in the video_input directory, change this if you have different file types
    video_names = [f for f in os.listdir('./video_input') if f.endswith('.mp4')]
    video_names = [v.replace('.mp4', '') for v in video_names]
    if len(video_names) == 0:
        raise AssertionError("No videos found in the video_input folder")
    
    video_option = ask_options("Which video do you want to analyze?", video_names)[0]
    video_selection = video_names[video_option - 1]
    
    # set model name and resolution 
    # list the directories under stardist_models folder
    model_names = [f for f in os.listdir('./stardist_models') if os.path.isdir(os.path.join('./stardist_models', f))]
    model_names = model_names + ['2D_versatile_fluo', '2D_paper_dsb2018', '2D_versatile_he']
    
    model_option = ask_options("Which Stardist model do you want to use?", model_names)[0] 
    model_name = model_names[model_option - 1]
    resolution = 1000
    
    tracking_options = ask_options("Which part do you want to run? (1, 2, ...)", ["Test", "Detection and classification", "Linking", "Interpolation", "Kalman filter & RTS smoother", "All of them"])
    # Set to false the following verbs if you want to import the data without running the process
    TEST = any(opt in tracking_options for opt in {1, 6})
    DETECT = any(opt in tracking_options for opt in {2, 6})
    LINK = any(opt in tracking_options for opt in {3, 6})
    INTERP = any(opt in tracking_options for opt in {4, 6})
    KALMAN = any(opt in tracking_options for opt in {5, 6})
    
    if INTERP:
        interpolation_options = ["linear", "nearest", "quadratic", "cubic"]
        interpolation_choice = ask_options("Which kernel do you want to use for interpolation? (1, 2, ...)", ["linear", "nearest", "quadratic", "cubic"])[0]
        interp_method = interpolation_options[interpolation_choice - 1]
    else:
        interp_method = None
    
    run_tracking_option = ask_options("Do you want to run the tracking or import the data?", ["Import the data", "Run the tracking"])[0]
    run_tracking_verb = run_tracking_option == 2
    
    
    save_plots = ask_yesno("Do you want to save the plots during the tracking?")
    animated_plot_results = ask_yesno("Do you want to plot animations during the tracking?")
    show_plots = ask_yesno("Do you want to see the plots during the tracking?")
    
    

    choices = {
        "video_selection": video_selection,
        "model_name": model_name,
        "resolution": resolution,
        "test": TEST,
        "detect": DETECT,
        "link": LINK,
        "interpolation": INTERP,
        "kalman": KALMAN,
        "save": save_plots,
        "show": show_plots,
        "animated": animated_plot_results
    }
    
    
        
    # select video to track and import parameters
    params = get_video_parameters(video_selection)
    globals().update(params)
    
    if crop_verb:
        pxDimension = petri_diameter/resolution # [mm/pixel]
    else:
        pxDimension = petri_diameter/(xmax - xmin) # [mm/pixel]

    res_path          = f'./tracking_results/{video_selection}/{model_name}/'
    os.makedirs(res_path, exist_ok = True)

    letter_labels = [f'{letter})' for letter in list(string.ascii_lowercase)]

    if not xmax - xmin == ymax - ymin:
        #raise AssertionError(f"The selected region is not square ({xmax - xmin} != {ymax - ymin})")
        print(f"The selected region is not square ({xmax - xmin} != {ymax - ymin})")

    print_recap_tracking(choices, model_name, resolution, interp_method)
    
    proceed = ask_yesno("Do you want to proceed with these choices?")
    if not proceed:
        user_message("Script aborted. Please restart the program.", "error")
        exit()
    else:
        user_message("Ok, let's go and proceed with the script.", "success")
        
    start = time.time()
    
    tracking = TrackingVideo(video_selection, model_name, video_source_path, xmin, ymin, xmax, ymax, resolution, n_instances, interp_method, res_path)
    fps = tracking.fps
    raw_detection_df, filtered_detection_df, raw_trajectory_df, interpolated_trajectory_df, kalman_rts_trajectories = None, None, None, None, None
    
    if 1:
        # plot the NN architecture
        #tf.keras.utils.plot_model(tracking.model.keras_model, to_file = f'./models/{model_name}/model.pdf', show_shapes=True, show_layer_names=True, rankdir='TB', expand_nested=False)

        # plot first and last frame with ROI full image
        full_img1 = tracking.get_frame(analysis_frames[0], crop_verb = False, resize_verb = True)
        full_img2 = tracking.get_frame(analysis_frames[-1], crop_verb = False, resize_verb = True)
        fig, (ax, ax1) = plt.subplots(1, 2, figsize = (12, 6))
        ax.imshow(full_img1)
        ax.add_artist(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, edgecolor='r', facecolor='none'))
        ax1.imshow(full_img2)
        ax1.add_artist(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, edgecolor='r', facecolor='none'))
        ax.set(title = f'Frame {analysis_frames[0]}', xticks = [], yticks = [])
        ax1.set(title = f'Frame {analysis_frames[-1]}', xticks = [], yticks = [])
        plt.tight_layout()
        if save_plots:
            plt.savefig(f'{res_path}/first_last_frame.pdf', bbox_inches = 'tight')
        if show_plots:
            plt.show()
        else:
            plt.close()
    
        # plot first and last frame with detection
        df1 = tracking.detect_instances(analysis_frames[:1], crop_verb, resize_verb = True, full_details = True)
        df2 = tracking.detect_instances(analysis_frames[-1:], crop_verb, resize_verb = True, full_details = True)
        if tracking.model.config.n_classes is not None:
            n_instances_detected1 = np.unique(df1['class_id'], return_counts = True)[1]
            n_instances_detected2 = np.unique(df2['class_id'], return_counts = True)[1]
        else:
            n_instances_detected1 = [len(df1)]
            n_instances_detected2 = [len(df2)]

        fig, (ax, ax1) = plt.subplots(1, 2, figsize = (10, 6), sharex=True, sharey=True)
        ax.imshow(tracking.get_frame(analysis_frames[0], crop_verb, True))
        ax1.imshow(tracking.get_frame(analysis_frames[-1], crop_verb, True))
        if not crop_verb:
            ax.add_artist(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, edgecolor='r', facecolor='none'))
            ax1.add_artist(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, edgecolor='r', facecolor='none'))
        if tracking.model.config.n_classes is not None:
            draw_polygons(polygons = np.array(df1.coord), points = np.array(df1.points), scores = np.array(df1.prob),\
                        colors = ['red' if i == 1 else 'blue' for i in df1.class_id], alpha = 1, show_dist = True, ax = ax)
            draw_polygons(polygons = np.array(df2.coord), points = np.array(df2.points), scores = np.array(df2.prob),\
                        colors = ['red' if i == 1 else 'blue' for i in df2.class_id], alpha = 1, show_dist = True, ax = ax1)
        else:
            draw_polygons(polygons = np.array(df1.coord), points = np.array(df1.points), scores = np.array(df1.prob),\
                        colors = None, alpha = 1, show_dist = True, ax = ax)
            draw_polygons(polygons = np.array(df2.coord), points = np.array(df2.points), scores = np.array(df2.prob),\
                        colors = None, alpha = 1, show_dist = True, ax = ax1)
        ax.text(0.0, 1.0, 'a)', transform=(ax.transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)), fontsize='medium', va='bottom')
        ax1.text(0.0, 1.0, 'b)', transform=(ax1.transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)), fontsize='medium', va='bottom')
        ax.set(xlim = (0, resolution), ylim = (resolution, 0), xticks = [], yticks = [])
        ax1.set(xlim = (0, resolution), ylim = (resolution, 0), xticks = [], yticks = [])
        if len(n_instances_detected1) > 1:
            ax.set_title(f'Frame {analysis_frames[0]} - {n_instances_detected1[0]} red -- {n_instances_detected1[1]} blue')
        else:	
            ax.set_title(f'Frame {analysis_frames[0]} - {n_instances_detected1[0]} droplets')
        if len(n_instances_detected2) > 1:
            ax1.set_title(f'Frame {analysis_frames[-1]} - {n_instances_detected2[0]} red -- {n_instances_detected2[1]} blue')
        else:
            ax1.set_title(f'Frame {analysis_frames[-1]} - {n_instances_detected2[0]} droplets')
        plt.tight_layout()
        if save_plots:
            plt.savefig(f'{res_path}/first_last_frame_detection.pdf', bbox_inches = 'tight')
        if show_plots:
            plt.show()
        else:
            plt.close()
     
    #########################################################################################
    #                              TEST DETECTION AND LINKING                               #
    #########################################################################################
        
    if TEST:
        test_frames = np.arange(0, 100, 1).astype(int) # change as needed
        test_df = tracking.detect_instances(test_frames, crop_verb = crop_verb, resize_verb = True, full_details = True)
        print('Pre filtering stats')
        counts_per_frame, err_frames, err_up, err_sub, max_n_of_consecutive_errs = get_n_errors(test_df, tracking.n_instances, test_frames)

        # filter detection
        test_df_filtered = test_df.loc[(test_df.x < xmax) & (test_df.x > xmin) & (test_df.y < ymax) & (test_df.y > ymin)]
        print('Post filtering stats')
        counts_per_frame_filtered, err_frames_filtered, err_up_filtered, err_sub_filtered, max_n_of_consecutive_errs_filtered = get_n_errors(test_df_filtered, tracking.n_instances, test_frames)    

        if 1:
            # show raw detection data filtering
            fig, ax = plt.subplots(2, 2, figsize = (8, 4))
            ax[0, 0].scatter(counts_per_frame.frame.unique(), counts_per_frame.counts, s=0.1)
            ax[0, 0].set(xlabel = 'Frame', ylabel = 'N of droplets', title = 'N of droplets per frame')
            ax[0, 1].scatter(test_df.frame, test_df.area, s=0.1)
            ax[0, 1].set(xlabel = 'Instance index', ylabel = r'area [$px^2$]', title = 'area of instances detected')
            ax[1, 0].scatter(test_df.area, test_df.eccentricity, s=0.1)
            ax[1, 0].set(xlabel = 'area [px^2]', ylabel='Eccentricity', title='area-eccentricity correlation')
            ax[1, 1].scatter(test_df.area, test_df.prob, s=0.1)
            ax[1, 1].set(xlabel = 'area [px^2]', ylabel='Probability', title='area-Probability correlation')
            for i, ax in enumerate(ax.flatten()):
                ax.text(0.0, 1.0, f'{letter_labels[i]}', transform=(ax.transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)), fontsize='medium', va='bottom')
            plt.tight_layout()
            if save_plots:
                plt.savefig(f'{res_path}/detection_quality.pdf', bbox_inches = 'tight')
            if show_plots:
                plt.show()
            else:
                plt.close()
        
            # show detection data after filtering
            fig, ax = plt.subplots(2, 2, figsize = (8, 4))
            ax[0, 0].scatter(counts_per_frame_filtered.frame.unique(), counts_per_frame_filtered.counts, s=0.1)
            ax[0, 0].set(xlabel = 'Frame', ylabel = 'N of droplets', title = 'N of droplets per frame')
            ax[0, 1].scatter(test_df_filtered.frame, test_df_filtered.r, s=0.1)
            ax[0, 1].set(xlabel = 'Instance index', ylabel = 'Radius [px]', title = 'Radius of instances detected')
            ax[1, 0].scatter(test_df_filtered.r, test_df_filtered.eccentricity, s=0.1)
            ax[1, 0].set(xlabel = 'Radius [px]', ylabel='Eccentricity', title='R-eccentricity correlation')
            ax[1, 1].scatter(test_df_filtered.r, test_df_filtered.prob, s=0.1)
            ax[1, 1].set(xlabel = 'Radius [px]', ylabel='Probability', title='R-Probability correlation')
            for i, ax in enumerate(ax.flatten()):
                ax.text(0.0, 1.0, f'{letter_labels[i]}', transform=(ax.transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)), fontsize='medium', va='bottom')
            plt.tight_layout()
            if save_plots:
                plt.savefig(f'{res_path}/detection_quality_filtered.pdf', bbox_inches = 'tight')
            if show_plots:
                plt.show()
            else:
                plt.close()
        
            # show an example of detection error after filtering
            if len(err_frames_filtered) > 0:
                err_frame = err_frames_filtered[0]
                df = test_df_filtered.loc[test_df_filtered.frame == err_frame]
                df1 = test_df.loc[test_df.frame == err_frame]

                fig, (ax, ax1, ax2) = plt.subplots(1, 3, figsize = (13, 6), sharex=True, sharey=True)
                ax.imshow(tracking.get_frame(err_frame, crop_verb, True))
                for i in range(len(df)):
                    c = plt.Circle((df.x.values[i], df.y.values[i]), df.r.values[i], color = 'r', fill = False)
                    ax.add_patch(c)
                ax.set_title(f'Filtered -- {len(df)} instances')
                ax1.imshow(tracking.get_frame(err_frame, crop_verb, True))
                for i in range(len(df1)):
                    c = plt.Circle((df1.x.values[i], df1.y.values[i]), df1.r.values[i], color = 'r', fill = False)
                    ax1.add_patch(c)
                ax1.set_title(f'Raw -- {len(df1)} instances')
                ax2.imshow(tracking.get_frame(err_frame, crop_verb, True))
                ax2.set_title(f'Original -- frame {err_frame}')
                ax.set(xlim = (0, resolution), ylim = (0, resolution))
                ax.text(0.0, 1.0, 'a)', transform=(ax.transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)), fontsize='medium', va='bottom')
                ax1.text(0.0, 1.0, 'b)', transform=(ax1.transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)), fontsize='medium', va='bottom')
                ax2.text(0.0, 1.0, 'c)', transform=(ax2.transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)), fontsize='medium', va='bottom')
                if save_plots:
                    plt.savefig(f'{res_path}/error_example.pdf', bbox_inches = 'tight')
                if show_plots:
                    plt.show()
                else:
                    plt.close()
            else:
                print('No errors detected in the test frames after filtering.')
                
                
            # show animation of detections for the first 100 test frames
            if animated_plot_results:
                fig, ax = fig, ax = plt.subplots(1, 1, figsize = (10, 10))
                anim_running = True
                def update_graph(frame):
                    ax.clear()
                    df = test_df_filtered.loc[test_df_filtered.frame == frame]
                    ax.imshow(tracking.get_frame(frame, crop_verb, True))
                    if tracking.model.config.n_classes is not None:
                        draw_polygons(polygons = np.array(df.coord), points = np.array(df.points), scores = np.array(df.prob),\
                                        colors = ['red' if i == 1 else 'blue' for i in df.class_id], alpha = 1, show_dist = True, ax = ax)
                    else:
                        draw_polygons(polygons = np.array(df.coord), points = np.array(df.points), scores = np.array(df.prob),\
                                        colors = None, alpha = 1, show_dist = True, ax = ax)
                    title = ax.set_title(f'{video_selection} Tracking -- frame = {frame}')
                    ax.set(xlabel = 'X [px]', ylabel = 'Y [px]')
                    return ax

                title = ax.set_title(f'{video_selection} Tracking -- frame = {analysis_frames[0]}')
                ax.set(xlabel = 'X [px]', ylabel = 'Y [px]')
                df = test_df_filtered.loc[test_df_filtered.frame == analysis_frames[0]]
                ax.imshow(tracking.get_frame(analysis_frames[0], crop_verb, True))
                if tracking.model.config.n_classes is not None:
                    draw_polygons(polygons = np.array(df.coord), points = np.array(df.points), scores = np.array(df.prob),\
                                colors = ['red' if i == 1 else 'blue' for i in df.class_id], alpha = 1, show_dist = True, ax = ax)
                else:
                    draw_polygons(polygons = np.array(df.coord), points = np.array(df.points), scores = np.array(df.prob),\
                                  colors = None, alpha = 1, show_dist = True, ax = ax)
                fig.canvas.mpl_connect('button_press_event', onClick)
                ani = matplotlib.animation.FuncAnimation(fig, update_graph, test_df_filtered.frame.unique()[:100], interval = 5, blit=False)
                writer = matplotlib.animation.FFMpegWriter(fps = 10, metadata = dict(artist='Matteo Scandola'), extra_args=['-vcodec', 'libx264'])
                if save_plots:
                    ani.save(f'./{res_path}/tracking_video_test.mp4', writer=writer, dpi = 500)
                plt.close()


        
        raw_trajectory_df = tracking.linking_detection(test_df_filtered, cutoff = 150, max_frame_gap = max_n_of_consecutive_errs_filtered + 1, min_trajectory_length = 1)

        # show the trajectories of the first 5 instances
        if 1:
            fig, ax = fig, ax = plt.subplots(1, 1, figsize = (10, 10))
            for i in range(5):
                df = raw_trajectory_df.loc[raw_trajectory_df.particle == i]
                ax.plot(df.x, df.y, 'o-')
            ax.set(xlabel = 'X [px]', ylabel = 'Y [px]', title = 'Trajectory of particles')
            ax.set(xlim = (0, resolution), ylim = (resolution, 0))
            if save_plots:
                plt.savefig(f'{res_path}/test_trajectories.pdf', bbox_inches = 'tight')
            if show_plots:
                plt.show()
            else:
                plt.close()


    #########################################################################################
    #                             DETECTION AND FILTERING RAW DATA                          #
    #########################################################################################
    if DETECT:
        if run_tracking_verb:
            raw_detection_df = tracking.detect_instances(analysis_frames, crop_verb = crop_verb, resize_verb = True, full_details = False)
            raw_detection_df.to_parquet(tracking.res_path + f'raw_detection_{analysis_frames[0]}_{analysis_frames[-1]}.parquet')
        else:
            print('Loading raw detections...')
            raw_detection_df = pd.read_parquet(tracking.res_path + f'raw_detection_{analysis_frames[0]}_{analysis_frames[-1]}.parquet')

        # if necessary, filter raw detection data in some way 
        filtered_detection_df = raw_detection_df.loc[(raw_detection_df.x < xmax) & (raw_detection_df.x > xmin) & (raw_detection_df.y < ymax) & (raw_detection_df.y > ymin)]
        
        # to save ram
        if raw_detection_df is not None: del raw_detection_df

        print("Pre filtering stats")
        counts_per_frame, err_frames, err_up, err_sub, max_n_of_consecutive_errs = get_n_errors(raw_detection_df, tracking.n_instances, analysis_frames)
        print("Post filtering stats")    
        counts_per_frame_filtered, err_frames_filtered, err_up_filtered, err_sub_filtered, max_n_of_consecutive_errs_filtered = get_n_errors(filtered_detection_df, tracking.n_instances, analysis_frames)    

        if 1:
            fig, ax = plt.subplots(2, 2, figsize = (8, 4))
            ax[0, 0].scatter(counts_per_frame.frame, counts_per_frame.counts, s=0.1)
            ax[0, 0].set(xlabel = 'Frame', ylabel = 'N of droplets', title = 'N of droplets per frame')
            ax[0, 1].scatter(raw_detection_df.frame, raw_detection_df.area, s=0.1)
            ax[0, 1].set(xlabel = 'Instance index', ylabel = r'Area [$px^2$]', title = 'Radius of instances detected')
            ax[1, 0].scatter(raw_detection_df.area, raw_detection_df.eccentricity, s=0.1)
            ax[1, 0].set(xlabel = r'Area [$px^2$]', ylabel='Eccentricity', title='Area-eccentricity correlation')
            ax[1, 1].scatter(raw_detection_df.area, raw_detection_df.prob, s=0.1)
            ax[1, 1].set(xlabel = r'Area [$px^2$]', ylabel='Probability', title='Area-Probability correlation')
            for i, ax in enumerate(ax.flatten()):
                ax.text(0.0, 1.0, f'{letter_labels[i]}', transform=(ax.transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)), fontsize='medium', va='bottom')
            plt.tight_layout()
            plt.savefig(tracking.res_path + f'raw_instances_{analysis_frames[0]}_{analysis_frames[-1]}.png', dpi = 500)
            if show_plots:
                plt.show()
            else:
                plt.close()
                
            fig, ax = plt.subplots(2, 2, figsize = (8, 4))
            ax[0, 0].scatter(counts_per_frame_filtered.frame, counts_per_frame_filtered.counts, s=0.1)
            ax[0, 0].set(xlabel = 'Frame', ylabel = 'N of droplets', title = 'N of droplets per frame')
            ax[0, 1].scatter(filtered_detection_df.frame, filtered_detection_df.area, s=0.1)
            ax[0, 1].set(xlabel = 'Instance index', ylabel = r'Area [$px^2$]', title = 'Radius of instances detected')
            ax[1, 0].scatter(filtered_detection_df.area, filtered_detection_df.eccentricity, s=0.1)
            ax[1, 0].set(xlabel = r'Area [$px^2$]', ylabel='Eccentricity', title='Area-eccentricity correlation')
            ax[1, 1].scatter(filtered_detection_df.area, filtered_detection_df.prob, s=0.1)
            ax[1, 1].set(xlabel = r'Area [$px^2$]', ylabel='Probability', title='Area-Probability correlation')
            for i, ax in enumerate(ax.flatten()):
                ax.text(0.0, 1.0, f'{letter_labels[i]}', transform=(ax.transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)), fontsize='medium', va='bottom')
            plt.tight_layout()
            plt.savefig(tracking.res_path + f'filtered_instances_{analysis_frames[0]}_{analysis_frames[-1]}.png', dpi = 500)
            if show_plots:
                plt.show()
            else:
                plt.close()


    #########################################################################################
    #                         LINK FILTERED DETECTIONS INTO TRAJECTORIES                    #
    #########################################################################################

    if LINK:
        if run_tracking_verb:
            
            if raw_detection_df is None:
                print('Loading raw detections...')
                raw_detection_df = pd.read_parquet(tracking.res_path + f'raw_detection_{analysis_frames[0]}_{analysis_frames[-1]}.parquet')
                # if necessary, filter raw detection data in some way 
                filtered_detection_df = raw_detection_df.loc[(raw_detection_df.x < xmax) & (raw_detection_df.x > xmin) & (raw_detection_df.y < ymax) & (raw_detection_df.y > ymin)]
                counts_per_frame_filtered, err_frames_filtered, err_up_filtered, err_sub_filtered, max_n_of_consecutive_errs_filtered = get_n_errors(filtered_detection_df, tracking.n_instances, analysis_frames) 
                del raw_detection_df
                
            
            raw_trajectory_df = tracking.linking_detection(filtered_detection_df, cutoff = 150, max_frame_gap = max_n_of_consecutive_errs_filtered + 1, min_trajectory_length = 5000)
            raw_trajectory_df.to_parquet(res_path + f'raw_tracking_{analysis_frames[0]}_{analysis_frames[-1]}.parquet', index = False)
        else:
            print('Loading linked trajectories...')
            raw_trajectory_df = pd.read_parquet(res_path + f'raw_tracking_{analysis_frames[0]}_{analysis_frames[-1]}.parquet')

        # save ram
        if filtered_detection_df is not None: del filtered_detection_df

        print("Post linking stats")
        counts_per_frame, err_frames, err_up, err_sub, max_n_of_consecutive_errs = get_n_errors(raw_trajectory_df, tracking.n_instances, analysis_frames)


        # check if class_id is preserved in multiclass detection
        if tracking.model.config.n_classes is not None:    
            if len(np.where(raw_trajectory_df.groupby('particle').class_id.nunique().values != 1)[0]) == 0:
                print('Class_id is preserved')   
            else:
                print('Class_id is not preserved')
                # fix class_id error, improve if necessary
                err_droplet_ids = np.where(raw_trajectory_df.groupby('particle').class_id.nunique().values != 1)[0]
                for err_droplet_id in err_droplet_ids:
                    n_frames_1 = len(raw_trajectory_df.loc[(raw_trajectory_df.particle == err_droplet_id) & (raw_trajectory_df.class_id == 1)])
                    n_frames_2 = len(raw_trajectory_df.loc[(raw_trajectory_df.particle == err_droplet_id) & (raw_trajectory_df.class_id == 2)])
                    print(f'Particle {err_droplet_id} has {n_frames_1} frames in class 1 and {n_frames_2} frames in class 2')
                    if n_frames_1 > n_frames_2:
                        raw_trajectory_df.loc[(raw_trajectory_df.particle == err_droplet_id) & (raw_trajectory_df.class_id == 2), 'class_id'] = 1
                    else:
                        raw_trajectory_df.loc[(raw_trajectory_df.particle == err_droplet_id) & (raw_trajectory_df.class_id == 1), 'class_id'] = 2
                    print(f'Particle {err_droplet_id} has been fixed')


        if 1:
            fig, ax = plt.subplots(2, 2, figsize = (8, 4))
            ax[0, 0].scatter(counts_per_frame.frame, counts_per_frame.counts, s=0.1)
            ax[0, 0].set(xlabel = 'Frame', ylabel = 'N of droplets', title = 'N of droplets per frame')
            ax[0, 1].scatter(raw_trajectory_df.frame, raw_trajectory_df.r, s=0.1)
            ax[0, 1].set(xlabel = 'Instance index', ylabel = 'Radius [px]', title = 'Radius of instances detected')
            ax[1, 0].scatter(raw_trajectory_df.r, raw_trajectory_df.eccentricity, s=0.1)
            ax[1, 0].set(xlabel = 'Radius [px]', ylabel='Eccentricity', title = 'R-eccentricity correlation')
            ax[1, 1].scatter(raw_trajectory_df.r, raw_trajectory_df.prob, s=0.1)
            ax[1, 1].set(xlabel = 'Radius [px]', ylabel='Probability', title = 'R-Probability correlation')
            for i, ax in enumerate(ax.flatten()):
                ax.text(0.0, 1.0, f'{letter_labels[i]}', transform=(ax.transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)), fontsize='medium', va='bottom')
            plt.tight_layout()
            plt.savefig(tracking.res_path + f'raw_tracking_{analysis_frames[0]}_{analysis_frames[-1]}.png', dpi = 500)
            if show_plots:
                plt.show()
            else:
                plt.close()
                
    #########################################################################################
    #                                   POST PROCESSING                                     #
    #########################################################################################


    if INTERP:
        if run_tracking_verb:
            if raw_trajectory_df is None:
                print('Loading linked trajectories...')
                raw_trajectory_df = pd.read_parquet(res_path + f'raw_tracking_{analysis_frames[0]}_{analysis_frames[-1]}.parquet')
                
            interpolated_trajectory_df = tracking.interp_raw_trajectory(raw_trajectory_df)
            interpolated_trajectory_df.to_parquet(res_path + f'interpolated_tracking_{analysis_frames[0]}_{analysis_frames[-1]}.parquet', index=False)
        else:
            print('Loading interpolated trajectories...')
            interpolated_trajectory_df = pd.read_parquet(res_path + f'interpolated_tracking_{analysis_frames[0]}_{analysis_frames[-1]}.parquet')
        
        # save ram
        if raw_trajectory_df is not None: del raw_trajectory_df

        counts_per_frame, err_frames, err_up, err_sub, max_n_of_consecutive_errs = get_n_errors(interpolated_trajectory_df, tracking.n_instances, analysis_frames)

        if 1:
            fig, ax = plt.subplots(2, 2, figsize = (8, 4))
            ax[0, 0].scatter(counts_per_frame.frame, counts_per_frame.counts, s=0.1)
            ax[0, 0].set(xlabel = 'Frame', ylabel = 'N of droplets', title = 'N of droplets per frame')
            ax[0, 1].scatter(interpolated_trajectory_df.frame, interpolated_trajectory_df.r, s=0.1)
            ax[0, 1].set(xlabel = 'Instance index', ylabel = 'Radius [px]', title = 'Radius of instances detected')
            ax[1, 0].scatter(interpolated_trajectory_df.r, interpolated_trajectory_df.eccentricity, s=0.1)
            ax[1, 0].set(xlabel = 'Radius [px]', ylabel='Eccentricity', title = 'R-eccentricity correlation')
            ax[1, 1].scatter(interpolated_trajectory_df.r, interpolated_trajectory_df.prob, s=0.1)
            ax[1, 1].set(xlabel = 'Radius [px]', ylabel='Probability', title = 'R-Probability correlation')
            for i, ax in enumerate(ax.flatten()):
                ax.text(0.0, 1.0, f'{letter_labels[i]}', transform=(ax.transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)), fontsize='medium', va='bottom')
            plt.tight_layout()
            plt.savefig(tracking.res_path + f'interpolated_tracking_{analysis_frames[0]}_{analysis_frames[-1]}.png', dpi = 500)
            if show_plots:
                plt.show()
            else:
                plt.close()
    
    if KALMAN:
        if run_tracking_verb:
            
            if interpolated_trajectory_df is None:
                print('Loading interpolated trajectories...')
                interpolated_trajectory_df = pd.read_parquet(res_path + f'interpolated_tracking_{analysis_frames[0]}_{analysis_frames[-1]}.parquet')
            
            subsample_factor = 1 # subsample the trajectory if needed
            kalman_rts_trajectories = tracking.kalman_filter_full(interpolated_trajectory_df, a = 1e-3, b = 2., measurement_sigma = 0.01, order = 2, cov_factor = .01, q = 2, subsample_factor=subsample_factor)
            if tracking.model.config.n_classes is not None:
                kalman_rts_trajectories['class_id'] = kalman_rts_trajectories['class_id'].astype(int)
            kalman_rts_trajectories.to_parquet(tracking.res_path + f'trajectories_kalman_rts_{analysis_frames[0]}_{analysis_frames[-1]}_subsample_factor_{subsample_factor}.parquet', engine='pyarrow')
        else:
            print('Loading Kalman filtered trajectories...')
            kalman_rts_trajectories = pd.read_parquet(tracking.res_path + f'trajectories_kalman_rts_{analysis_frames[0]}_{analysis_frames[-1]}_subsample_factor_{subsample_factor}.parquet', engine='pyarrow')

        # save ram
        if interpolated_trajectory_df is not None:
            df = interpolated_trajectory_df.loc[interpolated_trajectory_df.particle == 0]
            nFrames = len(interpolated_trajectory_df.frame.unique())
            del interpolated_trajectory_df
            
    
    end = time.time()
    minutes = int((end - start) // 60)
    user_message(f"Tracking script completed in {minutes} min", "succes")
        
if __name__ == "__main__":
    main()