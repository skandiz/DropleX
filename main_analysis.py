import os
import json
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({
	'font.size': 15,               # General font size
	'axes.titlesize': 14,          # Title font size
	'axes.labelsize': 12,          # Axis label font size
	'legend.fontsize': 10,         # Legend font size
	'xtick.labelsize': 10,         # X-axis label size
	'ytick.labelsize': 10          # Y-axis label size
})       
from matplotlib.transforms import ScaledTranslation

from analysis_utils import ask_options, ask_yesno, user_message, print_recap_analysis, get_analysis_parameters, get_video_properties, create_masks, compute_kinematics, compute_properties, compute_windowed_analysis, generate_plot_styles, create_directories, get_frame

def main():
    print("\nWelcome to DropleX, the Python tool for tracking and analysis of active particles from videos! \nLet's start by selecting the trajectory you want to analyze and the type of analyses you want to run. \nPlease follow the instructions below. \n")

    trajectory_names = list(json.load(open("./analysis_config.json", "r")).keys())
    if len(trajectory_names) == 0:
        raise AssertionError("No trajectories found in the analysis_config.json file. Please add the paths to the trajectories you want to analyze.")
    
    trajectory_option = ask_options("Which trajectory do you want to analyze?", trajectory_names)[0]

    analysis_options = ask_options("Which analyses do you want to run? (1, 2, ...)", ["Order parameters analysis", "Shape analysis", "Time Averaged Mean Squared Displacement analysis", "Speed distribution analysis", "Turning angles distribution analysis", "Velocity Autocovariance analysis", "Dimer distribution analysis", "All of them"])
    run_analysis_option = ask_options("Do you want to run the analysis or import the data?", ["Import the data", "Run the analysis"])[0]
    run_analysis_verb = run_analysis_option == 2
    
    save_plots = ask_yesno("Do you want to save the plots during the analysis?")
    animated_plot_results = ask_yesno("Do you want to plot animations during the analysis?")
    show_plots = ask_yesno("Do you want to see the plots during the analysis?")
    
    ORDER = any(opt in analysis_options for opt in {1, 8})
    SHAPE = any(opt in analysis_options for opt in {2, 8})
    TAMSD = any(opt in analysis_options for opt in {3, 8})
    SPEED = any(opt in analysis_options for opt in {4, 8})
    TURNING = any(opt in analysis_options for opt in {5, 8})
    VACF = any(opt in analysis_options for opt in {6, 8})
    DIMER = any(opt in analysis_options for opt in {7, 8})
    
    choices = {
        "video_selection": trajectory_names[trajectory_option - 1],
        "order": ORDER,
        "shape": SHAPE,
        "tamsd": TAMSD,
        "speed": SPEED,
        "turning": TURNING,
        "vacf": VACF,
        "dimer": DIMER,
        "run": run_analysis_verb,
        "save": save_plots,
        "show": show_plots,
        "animated": animated_plot_results
    }
        
    # Load analysis parameters
    params = get_analysis_parameters(trajectory_names[trajectory_option - 1], "./analysis_config.json")

    params['folder_names'] = ['dimer_analysis', 'tamsd_analysis', 'order_analysis', 'shape_analysis', 'speed_analysis', 'turning_angles_analysis', 'vacf_analysis']
    params['dimension_units'] = 'mm'
    params['speed_units'] = 'mm/s'
    params['video_selection'] = trajectory_names[trajectory_option - 1]
    params['window_length'] = 100 # [s]
    params['stride_length'] = 10 # [s]
    params['fps'] = int(params['video_fps']/params['subsample_factor'])
    params['frames_stages'] = np.array(params['stages_seconds'])*params['fps']
    trajectories = pd.read_parquet(params['trajectory_path'], engine='pyarrow', columns=['x', 'y', 'r', 'eccentricity', 'particle', 'frame', 'class_id'])
    trajectories.set_index('frame', inplace = True, drop = False)
    # Adjust frame numbering
    if trajectories.frame.min() > 0: trajectories.frame = trajectories.frame - trajectories.frame.min()
    frames = trajectories.frame.unique().astype(int)
    params['n_frames'] = len(frames)

    # Get red and blue particle indices
    params['red_particle_idx'] = np.sort(trajectories.loc[trajectories.class_id == 1].particle.unique())
    params['blue_particle_idx'] = np.sort(trajectories.loc[trajectories.class_id == 2].particle.unique())

    if params['crop_verb']:
       params['pxDimension'] = params['petri_diameter']/params['resolution'] # [mm/pixel]
    else:
        if params['video_selection'] in ['1b_&_1r_1', '1b_&_1r_2', '1b_&_1r_3']:
            pxDimension_blue = params['petri_diameter']/(params['xmax_b'] - params['xmin_b'])
            pxDimension_red = params['petri_diameter']/(params['xmax_r'] - params['xmin_r'])
            params['pxDimension'] = (pxDimension_blue + pxDimension_red)/2
        else:
            params['pxDimension'] = params['petri_diameter']/(params['xmax'] - params['xmin']) # [mm/pixel]

    params['red_mask'], params['colors'] = create_masks(params['n_particles'], params['red_particle_idx'])
    params['startFrames'], params['window_center_sec'], params['endFrames'], params['n_windows'], params['n_stages'], params['steps_plot'] = compute_windowed_analysis(frames, params['fps'], params['window_length'], params['stride_length'], params['frames_stages'])
    params['shades_of_blue'], params['default_kwargs_blue'], params['shades_of_red'], params['default_kwargs_red'], params['letter_labels'], params['stages_shades'] = generate_plot_styles(params['n_stages'])

    if params['video_selection'] in ['1b_&_1r_1', '1b_&_1r_2', '1b_&_1r_3']:
        video_blue, video_red, video_data = get_video_properties(params['video_selection'], video_source_path_blue = params['video_source_path_blue'], video_source_path_red = params['video_source_path_blue'])
    else:
        video, video_data = get_video_properties(params['video_selection'], video_source_path = params['video_source_path'])
        
    params = create_directories(params)
    
    print_recap_analysis(choices, params)
    
    proceed = ask_yesno("Do you want to proceed with these choices?")
    if not proceed:
        user_message("Analysis aborted. Please restart the program.", "error")
        exit()
    else:
        user_message("Ok, let's go and proceed with the analysis.", "success")    
    
    start = time.time()
    positions = trajectories.loc[:, ['x', 'y']].values.reshape(params['n_frames'], params['n_particles'], 2)
    velocities, accelerations = compute_kinematics(trajectories, params['fps'], params['n_frames'], params['n_particles'])
    orientations, radii, eccentricity = compute_properties(trajectories, params['n_frames'], params['n_particles'], velocities)
    
    if 1:
        if params['video_selection'] in ['1b_&_1r_1', '1b_&_1r_2', '1b_&_1r_3']:
            fig, axs = plt.subplots(2, 5, figsize = (15, 5), sharex=True, sharey=True)
            for i in range(len(params['steps_plot'])):
                #hours, remainder = divmod(int(params['frames_stages'][i]/params['fps']), 3600)
                #minutes, seconds = divmod(remainder, 60)
                axs[0, i].imshow(get_frame(video_blue, params['initial_offset_b'] + params['frames_stages'][i]*params['subsample_factor'], params['xmin_b'], params['ymin_b'], params['xmax_b'], params['ymax_b'], params['resolution'], params['crop_verb']))
                axs[0, i].set(title = f"Stage {i + 1}", xticks = [], yticks = [])
                axs[0, i].text(0.0, 1.0, f"{params['letter_labels'][i]}", transform=(axs[0, i].transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)), fontsize='medium', va='bottom')
                
                axs[1, i].imshow(get_frame(video_red, params['initial_offset_r'] + params['frames_stages'][i]*params['subsample_factor'], params['xmin_r'], params['ymin_r'], params['xmax_r'], params['ymax_r'], params['resolution'], params['crop_verb']))
                axs[1, i].text(0.0, 1.0, f"{params['letter_labels'][i+5]}", transform=(axs[1, i].transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)), fontsize='medium', va='bottom')
            plt.tight_layout()
            if save_plots:
                plt.savefig(f"{params['res_path']}/stages_5.png", bbox_inches='tight')
                plt.savefig(f"{params['pdf_res_path']}/stages_5.pdf", bbox_inches='tight')
            if show_plots:
                plt.show()
            else:
                plt.close()
                
            df_b = trajectories.loc[(trajectories.frame == 0) & (trajectories.particle == 0)]
            df_b1 = trajectories.loc[(trajectories.frame == params['fps']) & (trajectories.particle == 0)]
            df_b2 = trajectories.loc[(trajectories.frame == n_frames - 1) & (trajectories.particle == 0)]

            df_r = trajectories.loc[(trajectories.frame == 0) & (trajectories.particle == 1)]
            df_r1 = trajectories.loc[(trajectories.frame == params['fps']) & (trajectories.particle == 1)]
            df_r2 = trajectories.loc[(trajectories.frame == n_frames - 1) & (trajectories.particle == 1)]
            
            fig, axs = plt.subplots(2, 2, figsize = (10, 10), sharex=True, sharey=True)
            axs[0, 0].imshow(get_frame(video_blue, params['initial_offset_b'], params['xmin_b'], params['ymin_b'], params['xmax_b'], params['ymax_b'], params['resolution'], params['crop_verb']))
            axs[0, 0].add_artist(plt.Rectangle((params['xmin_b'], params['ymin_b']), params['xmax_b']-params['xmin_b'], params['ymax_b']-params['ymin_b'], edgecolor='b', facecolor='none'))
            axs[0, 0].add_artist(plt.Circle((df_b.x.values[0], df_b.y.values[0]), df_b.r.values[0], edgecolor='b', facecolor='none'))
            axs[0, 1].imshow(get_frame(video_blue, params['initial_offset_b'] + n_frames, params['xmin_b'], params['ymin_b'], params['xmax_b'], params['ymax_b'], params['resolution'], params['crop_verb']))
            axs[0, 1].add_artist(plt.Rectangle((params['xmin_b'], params['ymin_b']), params['xmax_b']-params['xmin_b'], params['ymax_b']-params['ymin_b'], edgecolor='b', facecolor='none'))
            axs[0, 1].add_artist(plt.Circle((df_b2.x.values[0], df_b2.y.values[0]), df_b2.r.values[0], edgecolor='b', facecolor='none'))
        
            axs[1, 0].imshow(get_frame(video_red, params['initial_offset_r'], params['xmin_r'], params['ymin_r'], params['xmax_r'], params['ymax_r'], params['resolution'], params['crop_verb']))
            axs[1, 0].add_artist(plt.Rectangle((params['xmin_r'], params['ymin_r']), params['xmax_r']-params['xmin_r'], params['ymax_r']-params['ymin_r'], edgecolor='r', facecolor='none'))
            axs[1, 0].add_artist(plt.Circle((df_r.x.values[0], df_r.y.values[0]), df_r.r.values[0], edgecolor='r', facecolor='none'))
            axs[1, 1].imshow(get_frame(video_red, params['initial_offset_r'] + n_frames, params['xmin_r'], params['ymin_r'], params['xmax_r'], params['ymax_r'], params['resolution'], params['crop_verb']))
            axs[1, 1].add_artist(plt.Rectangle((params['xmin_r'], params['ymin_r']), params['xmax_r']-params['xmin_r'], params['ymax_r']-params['ymin_r'], edgecolor='r', facecolor='none'))
            axs[1, 1].add_artist(plt.Circle((df_r2.x.values[0], df_r2.y.values[0]), df_r2.r.values[0], edgecolor='r', facecolor='none'))
            
            axs[0, 0].text(0.0, 1.0, 'a)', transform = (axs[0, 0].transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)), fontsize='medium', va='bottom')
            axs[0, 1].text(0.0, 1.0, 'b)', transform=(axs[0, 1].transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)), fontsize='medium', va='bottom')
            axs[1, 0].text(0.0, 1.0, 'c)', transform = (axs[1, 0].transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)), fontsize='medium', va='bottom')
            axs[1, 1].text(0.0, 1.0, 'd)', transform=(axs[1, 1].transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)), fontsize='medium', va='bottom')
            axs[0, 0].set(xlim = (0, params['resolution']), ylim = (params['resolution'], 0), xticks = [], yticks = [])
            plt.tight_layout()
            if save_plots:  
                plt.savefig(f"./{params['res_path']}/initial_final_frame.png", bbox_inches='tight', dpi = 300)
                plt.savefig(f"./{params['pdf_res_path']}/initial_final_frame.pdf", bbox_inches='tight')
            if show_plots:
                plt.show()
            else:
                plt.close()
                
            fig, axs = plt.subplots(2, 2, figsize = (10, 10), sharex = True, sharey = True)
            axs[0, 0].imshow(get_frame(video_blue, params['initial_offset_b'], params['xmin_b'], params['ymin_b'], params['xmax_b'], params['ymax_b'], params['resolution'], params['crop_verb']))
            axs[0, 0].add_artist(plt.Circle((df_b.x.values[0], df_b.y.values[0]), df_b.r.values[0], edgecolor='b', facecolor='none'))
            axs[0, 0].text(df_b.x.values[0], df_b.y.values[0], df_b.particle.values[0], color = 'black', fontsize = 6, horizontalalignment='center', verticalalignment='center')
            axs[0, 1].imshow(get_frame(video_blue, params['initial_offset_b'] + params['fps'], params['xmin_b'], params['ymin_b'], params['xmax_b'], params['ymax_b'], params['resolution'], params['crop_verb']))
            axs[0, 1].add_artist(plt.Circle((df_b1.x.values[0], df_b1.y.values[0]), df_b1.r.values[0], edgecolor='b', facecolor='none'))
            axs[0, 1].text(df_b1.x.values[0], df_b1.y.values[0], df_b1.particle.values[0], color = 'black', fontsize = 6, horizontalalignment='center', verticalalignment='center')
                
            axs[1, 0].imshow(get_frame(video_red, params['initial_offset_r'], params['xmin_r'], params['ymin_r'], params['xmax_r'], params['ymax_r'], params['resolution'], params['crop_verb']))
            axs[1, 0].add_artist(plt.Circle((df_r.x.values[0], df_r.y.values[0]), df_r.r.values[0], edgecolor='r', facecolor='none'))
            axs[1, 0].text(df_r.x.values[0], df_r.y.values[0], df_r.particle.values[0], color = 'black', fontsize = 6, horizontalalignment='center', verticalalignment='center')
            
            axs[1, 1].imshow(get_frame(video_red, params['initial_offset_r'] + params['fps'], params['xmin_r'], params['ymin_r'], params['xmax_r'], params['ymax_r'], params['resolution'], params['crop_verb']))
            axs[1, 1].add_artist(plt.Circle((df_r1.x.values[0], df_r1.y.values[0]), df_r1.r.values[0], edgecolor='r', facecolor='none'))
            axs[1, 1].text(df_r1.x.values[0], df_r1.y.values[0], df_r1.particle.values[0], color = 'black', fontsize = 6, horizontalalignment='center', verticalalignment='center')
                
            axs[0, 0].text(0.0, 1.0, 'a)', transform=(axs[0, 0].transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)), fontsize='medium', va='bottom')
            axs[0, 1].text(0.0, 1.0, 'b)', transform=(axs[0, 1].transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)), fontsize='medium', va='bottom')
            axs[1, 0].text(0.0, 1.0, 'c)', transform=(axs[1, 0].transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)), fontsize='medium', va='bottom')
            axs[1, 1].text(0.0, 1.0, 'd)', transform=(axs[1, 1].transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)), fontsize='medium', va='bottom')
            axs[0, 0].set(xticks = [], yticks=[], title = 'T = 0 s')
            axs[0, 1].set(xticks=[], yticks=[], title = 'T = 1 s')
            if save_plots:
                plt.savefig(f"./{params['res_path']}/identity.png", bbox_inches='tight', dpi = 300)
                plt.savefig(f"./{params['pdf_res_path']}/identity.pdf", bbox_inches='tight')
            if show_plots:
                plt.show()
            else:
                plt.close()
            
        else:    
            fig, axs = plt.subplots(1, 5, figsize = (15, 5), sharex=True, sharey=True)
            for i in range(len(params['steps_plot'])):
                hours, remainder = divmod(int(params['frames_stages'][i]/params['fps']), 3600)
                minutes, seconds = divmod(remainder, 60)
                axs[i].imshow(get_frame(video, params['initial_offset'] + params['frames_stages'][i]*params['subsample_factor'], params['xmin'], params['ymin'], params['xmax'], params['ymax'], params['resolution'], params['crop_verb']))
                axs[i].set(title = f"Stage {i + 1} -- T = {hours:02d}:{minutes:02d}:{seconds:02d}", xticks = [], yticks = [])
                axs[i].text(0.0, 1.0, f"{params['letter_labels'][i]}", transform=(axs[i].transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)), fontsize='medium', va='bottom')
            plt.tight_layout()
            if save_plots:
                plt.savefig(f"{params['res_path']}/stages_5.png", bbox_inches='tight')
                plt.savefig(f"{params['pdf_res_path']}/stages_5.pdf", bbox_inches='tight')
            if show_plots:
                plt.show()
            else:
                plt.close()
                
            df = trajectories.loc[trajectories.frame == frames[0]]
            df1 = trajectories.loc[trajectories.frame == frames[params['fps']]]
            df2 = trajectories.loc[trajectories.frame == frames[-1]]
            
            fig, (ax, ax1) = plt.subplots(1, 2, figsize = (10, 5), sharex = True, sharey = True)
            ax.imshow(get_frame(video, params['initial_offset'] + frames[0]*params['subsample_factor'], params['xmin'], params['ymin'], params['xmax'], params['ymax'], params['resolution'], params['crop_verb']))
            for i in range(len(df)):
                ax.text(df.x.iloc[i], df.y.iloc[i], df.particle.iloc[i], color = 'black', fontsize = 6, horizontalalignment='center', verticalalignment='center')
                ax.add_artist(plt.Circle((df.x.iloc[i], df.y.iloc[i]), df.r.iloc[i], color = params['colors'][i], fill = False, linewidth=1))
            ax1.imshow(get_frame(video, params['initial_offset'] + frames[params['fps']]*params['subsample_factor'], params['xmin'], params['ymin'], params['xmax'], params['ymax'], params['resolution'], params['crop_verb']))
            for i in range(len(df1)):
                ax1.text(df1.x.iloc[i], df1.y.iloc[i], df1.particle.iloc[i], color = 'black', fontsize = 6, horizontalalignment='center', verticalalignment='center')
                ax1.add_artist(plt.Circle((df1.x.iloc[i], df1.y.iloc[i]), df1.r.iloc[i], color = params['colors'][i], fill = False, linewidth=1))
            ax.text(0.0, 1.0, 'a)', transform=(ax.transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)), fontsize='medium', va='bottom')
            ax1.text(0.0, 1.0, 'b)', transform=(ax1.transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)), fontsize='medium', va='bottom')
            ax.set(xticks=[], yticks=[], title = 'T = 0 s')
            ax1.set(xticks=[], yticks=[], title = 'T = 1 s')
            if save_plots:
                plt.savefig(f"./{params['res_path']}/identity.png", bbox_inches='tight', dpi = 300)
                plt.savefig(f"./{params['pdf_res_path']}/identity.pdf", bbox_inches='tight')
            if show_plots:
                plt.show()
            else:
                plt.close()

            
            fig, (ax, ax1) = plt.subplots(1, 2, figsize = (10, 5))
            ax.imshow(get_frame(video, params['initial_offset'] + frames[0]*params['subsample_factor'], params['xmin'], params['ymin'], params['xmax'], params['ymax'], params['resolution'], params['crop_verb']))
            for i in range(len(df)):
                if i in params['red_particle_idx']:
                    ax.add_artist(plt.Circle((df.x.iloc[i], df.y.iloc[i]), df.r.iloc[i], color = 'red', fill = False))
                else:
                    ax.add_artist(plt.Circle((df.x.iloc[i], df.y.iloc[i]), df.r.iloc[i], color = 'blue', fill = False))
            if not params['crop_verb']:
                ax.add_artist(plt.Rectangle((params['xmin'], params['ymin']), params['xmax']-params['xmin'], params['ymax']-params['ymin'], edgecolor='k', facecolor='none'))
            
            ax1.imshow(get_frame(video, params['initial_offset'] + frames[-1]*params['subsample_factor'], params['xmin'], params['ymin'], params['xmax'], params['ymax'], params['resolution'], params['crop_verb']))
            for i in range(len(df2)):
                if i in params['red_particle_idx']:
                    ax1.add_artist(plt.Circle((df2.x.iloc[i], df2.y.iloc[i]), df2.r.iloc[i], color = 'red', fill = False))
                else:
                    ax1.add_artist(plt.Circle((df2.x.iloc[i], df2.y.iloc[i]), df2.r.iloc[i], color = 'blue', fill = False))
            if not params['crop_verb']:
                ax1.add_artist(plt.Rectangle((params['xmin'], params['ymin']), params['xmax']-params['xmin'], params['ymax']-params['ymin'], edgecolor='k', facecolor='none'))
            
            ax.text(0.0, 1.0, 'a)', transform=(ax.transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)), fontsize='medium', va='bottom')
            ax1.text(0.0, 1.0, 'b)', transform=(ax1.transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)), fontsize='medium', va='bottom')
            ax.set(xticks=[], yticks=[], title = 'Initial Frame')
            ax1.set(xticks=[], yticks=[], title = 'Final Frame')
            plt.tight_layout()
            if save_plots:
                plt.savefig(f"./{params['res_path']}/initial_final_frame.png", bbox_inches='tight', dpi = 300)
                plt.savefig(f"./{params['pdf_res_path']}/initial_final_frame.pdf", bbox_inches='tight')
            if show_plots:
                plt.show()
            else:
                plt.close()
    
    
    if ORDER:
        print("\n")
        user_message("Running Order parameters analysis...", "info")
        from analysis_modules.order_analysis import run_order_analysis
        velocity_polarization, hex_order = run_order_analysis(orientations, positions, radii, frames, params, video, save_plots, show_plots)
        velocity_pol_b, velocity_pol_r = velocity_polarization
        hex_order_real_wind_b, hex_order_real_wind_r, hex_order_real_wind = hex_order

    if SHAPE:
        print("\n")
        user_message("Running Shape analysis...", "info")
        from analysis_modules.shape_analysis import run_shape_analysis
        run_shape_analysis(trajectories, radii, eccentricity, frames, params, save_plots, show_plots)
    
    if TAMSD:
        print("\n")
        user_message("Running Time Averaged Mean Squared Displacement analysis...", "info")
        from analysis_modules.tamsd_analysis import run_tamsd_analysis
        EMSD_wind, pw_exp, maxLagtime_msd = run_tamsd_analysis(trajectories, frames, params, show_plots, save_plots, run_analysis_verb, animated_plot_results)
    else:
        EMSD_wind, pw_exp, maxLagtime_msd = [None, None], [None, None], None
        
    if SPEED:
        print("\n")
        user_message("Running Speed distribution analysis...", "info")
        from analysis_modules.speed_analysis import run_speed_analysis
        run_speed_analysis(trajectories, params, show_plots, save_plots, run_analysis_verb, animated_plot_results)
    
    if TURNING:
        print("\n")
        user_message("Running Turning angles distribution analysis...", "info")
        from analysis_modules.turning_angles_analysis import run_turning_analysis
        turn_angles, lorentzian_fit_results_wind, binning_info = run_turning_analysis(trajectories, frames, EMSD_wind, pw_exp, maxLagtime_msd, params, show_plots, save_plots, run_analysis_verb, animated_plot_results)
        turn_angles_b, turn_angles_r = turn_angles
        lorentzian_fit_results_wind_b, lorentzian_fit_results_wind_r = lorentzian_fit_results_wind
        turn_angles_bins, turn_angles_bin_centers, x_interval_for_fit_turn = binning_info
            
    if VACF:
        print("\n")
        user_message("Running Velocity Autocovariance analysis...", "info")
        from analysis_modules.vacf_analysis import run_vacf_analysis
        run_vacf_analysis(trajectories, params, show_plots, save_plots, run_analysis_verb, animated_plot_results)
    
    if DIMER:
        print("\n")
        user_message("Running Dimer distribution analysis...", "info")
        from analysis_modules.dimer_analysis import run_dimer_analysis
        dimer_distr_windowed, rbins, v_max = run_dimer_analysis(trajectories, radii, params, show_plots, save_plots, run_analysis_verb, animated_plot_results)
        dimer_distr_windowed_bb, dimer_distr_windowed_rr, dimer_distr_windowed_br, dimer_distr_windowed_rb = dimer_distr_windowed
        
    end = time.time()
    minutes = int((end - start) // 60)
    user_message(f"Analysis completed in {minutes} min", "succes")
    
if __name__ == "__main__":
    main()