import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({
	'font.size': 12,               # General font size
	'axes.titlesize': 14,          # Title font size
	'axes.labelsize': 12,          # Axis label font size
	'legend.fontsize': 10,         # Legend font size
	'xtick.labelsize': 10,
	'ytick.labelsize': 10})
from matplotlib.transforms import ScaledTranslation

def run_shape_analysis(trajectories, radii, eccentricity, frames, params, save_plots, show_plots):
    if len(params['blue_particle_idx']) > 0:
        mean_radius_b_wind = np.zeros((params['n_windows'], 2))
        eccentricity_blue_windowed = np.zeros((params['n_windows'], 2))
        for step in range(params['n_windows']):
            start = params['startFrames'][step] - frames[0]
            end = params['endFrames'][step] - frames[0]
            temp = np.mean(eccentricity[start:end], axis = 0)
            eccentricity_blue_windowed[step, 0] = np.mean(temp[~params['red_mask']])
            eccentricity_blue_windowed[step, 1] = np.std(temp[~params['red_mask']])

            temp = np.mean(radii[start:end], axis = 0) * params['pxDimension']
            mean_radius_b_wind[step, 0] = np.mean(temp[~params['red_mask']])
            mean_radius_b_wind[step, 1] = np.std(temp[~params['red_mask']])

    if len(params['red_particle_idx']) > 0:
        mean_radius_r_wind = np.zeros((params['n_windows'], 2))
        eccentricity_red_windowed = np.zeros((params['n_windows'], 2))
        for step in range(params['n_windows']):
            start = params['startFrames'][step] - frames[0]
            end = params['endFrames'][step] - frames[0]
            temp = np.mean(eccentricity[start:end], axis = 0)
            eccentricity_red_windowed[step, 0] = np.mean(temp[params['red_mask']])
            eccentricity_red_windowed[step, 1] = np.std(temp[params['red_mask']])

            temp = np.mean(radii[start:end], axis = 0) * params['pxDimension']
            mean_radius_r_wind[step, 0] = np.mean(temp[params['red_mask']])
            mean_radius_r_wind[step, 1] = np.std(temp[params['red_mask']])
    
    fig, ax = plt.subplots(1, 1,figsize = (8, 4))
    if len(params['blue_particle_idx']) > 0:
        ax.plot(params['window_center_sec'], eccentricity_blue_windowed[:, 0], color = 'b', alpha = 0.5)
        ax.fill_between(params['window_center_sec'], eccentricity_blue_windowed[:, 0] - 2/np.sqrt(len(params['blue_particle_idx'])) * eccentricity_blue_windowed[:, 1], eccentricity_blue_windowed[:, 0] + 2/np.sqrt(len(params['blue_particle_idx'])) * eccentricity_blue_windowed[:, 1], color = 'b', alpha = 0.2)
    if len(params['red_particle_idx']) > 0:
        ax.plot(params['window_center_sec'], eccentricity_red_windowed[:, 0], color = 'r', alpha = 0.5)
        ax.fill_between(params['window_center_sec'], eccentricity_red_windowed[:, 0] - 2/np.sqrt(len(params['red_particle_idx'])) * eccentricity_red_windowed[:, 1], eccentricity_red_windowed[:, 0] + 2/np.sqrt(len(params['red_particle_idx'])) * eccentricity_red_windowed[:, 1], color = 'r', alpha = 0.2)
    ax.set(xlabel = 'Window time [s]', ylabel = r'$\langle \epsilon \rangle$', title = 'Eccentricity evolution')
    for i, frame in enumerate(params['frames_stages']):
        ax.bar(frame/params['fps'], height = 2000, width = params['window_length'], bottom = -10, color = params['stages_shades'][i], alpha = 0.5, label = f"Stage {i + 1}")
    ax.set(xlabel = 'Window time [s]', ylabel = 'r [mm]', title = f"Droplets eccentricity of system {params['system_name']}")
    if params['video_selection'] in ['25b25r_lowconc_1', '25b25r_lowconc_2', '25b25r_lowconc_3', '25b25r_lowconc_5', '25b25r_lowconc_6']:
        ax.set(ylim =(0.1, 0.13),  xlim = (-200, 14000))
    elif params['video_selection'] in ['25b25r-1', '25b25r-2']:
        ax.set(ylim = (1.7, 2.5))
    elif params['video_selection'] in ['1b_&_1r_1', '1b_&_1r_2', '1b_&_1r_3']:
        ax.set(ylim = (0.3, .7))
    else:
        ax.set(ylim = (0, 0.5))
    ax.grid(linewidth = 0.2)
    ax.legend(loc = (0.57, 0.5), fontsize = 10)
    if save_plots:
        plt.savefig(f"{params['res_path']}/shape_analysis/mean_eccentricity_windowed_{params['n_stages']}_stages.png", bbox_inches='tight')
        plt.savefig(f"{params['pdf_res_path']}/shape_analysis/mean_eccentricity_windowed_{params['n_stages']}_stages.pdf", bbox_inches='tight')
    if show_plots:
        plt.show()
    else:
        plt.close()

    fig, ax = plt.subplots(1, 1, figsize = (8, 4))
    if len(params['blue_particle_idx']) > 0:
        ax.plot(params['window_center_sec'], mean_radius_b_wind[:, 0], 'b')
        ax.fill_between(params['window_center_sec'], mean_radius_b_wind[:, 0] - 2/np.sqrt(len(params['blue_particle_idx'])) * mean_radius_b_wind[:, 1], \
                        mean_radius_b_wind[:, 0] + 2/np.sqrt(len(params['blue_particle_idx'])) * mean_radius_b_wind[:, 1], color = 'b', alpha=0.5, facecolor='#00FFFF')
    if len(params['red_particle_idx']) > 0:
        ax.plot(params['window_center_sec'], mean_radius_r_wind[:, 0], 'r')
        ax.fill_between(params['window_center_sec'], mean_radius_r_wind[:, 0] - 2/np.sqrt(len(params['red_particle_idx'])) * mean_radius_r_wind[:, 1], \
                        mean_radius_r_wind[:, 0] + 2/np.sqrt(len(params['red_particle_idx'])) * mean_radius_r_wind[:, 1], color = 'r', alpha=0.5, facecolor='#FF5A52') 
    for i, frame in enumerate(params['frames_stages']):
        ax.bar(frame/params['fps'], height = 2000, width = params['window_length'], bottom = -10, color = params['stages_shades'][i], alpha = 0.5, label = f"Stage {i + 1}")
    ax.set(xlabel = 'Window time [s]', ylabel = 'r [mm]', title = f"Droplets radius of system {params['system_name']}")
    if params['video_selection'] in ['25b25r_lowconc_1', '25b25r_lowconc_2', '25b25r_lowconc_3', '25b25r_lowconc_5', '25b25r_lowconc_6']:
        ax.set(ylim = (1.4, 2), xlim = (-200, 14000))
    elif params['video_selection'] in ['25b25r-1', '25b25r-2']:
        ax.set(ylim = (1.7, 2.5))
    elif params['video_selection'] in ['1b_&_1r_1', '1b_&_1r_2', '1b_&_1r_3']:
        ax.set(ylim = (2, 3))
    else:
        ax.set(ylim = (1.2, 2.2))
    ax.grid(linewidth = 0.2)
    ax.legend(loc = (0.57, 0.5), fontsize = 10)
    if save_plots:
        plt.savefig(f"{params['res_path']}/shape_analysis/mean_radius_windowed_{params['n_stages']}_stages.png", bbox_inches='tight')
        plt.savefig(f"{params['pdf_res_path']}/shape_analysis/mean_radius_windowed_{params['n_stages']}_stages.pdf", bbox_inches='tight')
    if show_plots:
        plt.show()
    else:
        plt.close()
        
    try:
        orientation = trajectories.orientation.values.reshape(params['n_frames'], params['n_particles'])
        x1 = positions[:, :, 0] + np.cos(orientation) * 0.5 * trajectories.axis_minor_length.values.reshape(params['n_frames'], params['n_particles'])
        y1 = positions[:, :, 1] - np.sin(orientation) * 0.5 * trajectories.axis_minor_length.values.reshape(params['n_frames'], params['n_particles'])
        minor_axis_point = np.stack((x1, y1), axis=-1)

        x2 = positions[:, :, 0] - np.sin(orientation) * 0.5 * trajectories.axis_major_length.values.reshape(params['n_frames'], params['n_particles'])
        y2 = positions[:, :, 1] - np.cos(orientation) * 0.5 * trajectories.axis_major_length.values.reshape(params['n_frames'], params['n_particles'])
        major_axis_point = np.stack((x2, y2), axis=-1)

        axis_major_length = np.linalg.norm(major_axis_point - positions, axis = -1) * 2
        axis_minor_length = np.linalg.norm(minor_axis_point - positions, axis = -1) * 2
        angle = np.degrees(np.arctan2(major_axis_point[:, :, 1] - positions[:, :, 1], major_axis_point[:, :, 0] - positions[:, :, 0]))

        normalized_minor_axis_vector = (positions - minor_axis_point) / (np.linalg.norm(positions - minor_axis_point, axis = -1).reshape(params['n_frames'], params['n_particles'], 1))
        correlation_minor_axis_orientation = np.linalg.norm(orientations * normalized_minor_axis_vector, axis = 2)
        correlation_minor_axis_orientation_windowed = np.zeros((params['n_windows'], params['n_particles']))

        for step in range(params['n_windows']):
            start = params['startFrames'][step] - frames[0]
            end = params['endFrames'][step] - frames[0]
            correlation_minor_axis_orientation_windowed[step] = np.mean(correlation_minor_axis_orientation[start:end], axis = 0)
            
        if plot_verb:
            fig, ax = plt.subplots(1, 1, figsize = (8, 4))
            if len(params['blue_particle_idx']) > 0:
                ax.plot(params['window_center_sec'], np.mean(correlation_minor_axis_orientation_windowed[:, ~params['red_mask']], axis = 1), 'b', label = 'Blue droplet')
            if len(params['red_particle_idx']) > 0:
                ax.plot(params['window_center_sec'], np.mean(correlation_minor_axis_orientation_windowed[:, params['red_mask']], axis = 1), 'r', label = 'Red droplet')
            ax.grid()
            for i, frame in enumerate(params['frames_stages']):
                ax.bar(frame/params['fps'], height = 2000, width = params['window_length'], bottom = -10, color = params['stages_shades'][i], alpha = 0.5, label = f"Stage {i + 1}")
            ax.set(ylabel = 'K', ylim = (.6, 1), xlabel = 'Window Time [s]', title = f"Minor axis vector - orientation correlation of system {params['system_name']}")
            ax.legend(loc = (0.7, 0.5), fontsize = 10)
            if save_plots:
                plt.savefig(f"./{params['res_path']}/shape_analysis/minor_axis_orientation_correlation.png", bbox_inches='tight', dpi = 300)
                plt.savefig(f"./{params['pdf_res_path']}/shape_analysis/minor_axis_orientation_correlation.pdf", bbox_inches='tight')
            if show_plots:
                plt.show()
            else:
                plt.close()
    except:
        print('    No data for minor axis vs orientation correlation')