import numpy as np
np.seterr(invalid='ignore')
    
from tqdm import tqdm
import matplotlib.pyplot as plt
plt.rcParams.update({
	'font.size': 12,               # General font size
	'axes.titlesize': 14,          # Title font size
	'axes.labelsize': 12,          # Axis label font size
	'legend.fontsize': 10,         # Legend font size
	'xtick.labelsize': 10,
	'ytick.labelsize': 10})
from matplotlib.transforms import ScaledTranslation
from analysis_utils import compute_hex_order, get_neighbours_props, get_frame

def run_order_analysis(orientations, positions, radii, frames, params, video, save_plots, show_plots):
    if len(params['blue_particle_idx']) > 0:
        velocity_pol_b = np.zeros(params['n_windows'])
        for step in tqdm(range(params['n_windows']), desc = "    Computing blue droplets velocity polarization"):
            start = params['startFrames'][step]
            end = params['endFrames'][step]
            velocity_pol_b[step] = np.mean(np.linalg.norm(np.mean(orientations[start:end, ~params['red_mask']], axis = 1), axis = 1))

        res_blue = compute_hex_order(positions[:, ~params['red_mask']], np.mean(radii[:, ~params['red_mask']], axis = 1), description = "    Computing blue droplets hexatic order        ")
        hex_order_real_b, hex_order_img_b, n_of_neighbors_b, n_of_neighbors_b_std = res_blue[:, 0], res_blue[:, 1], res_blue[:, 2], res_blue[:, 3]
        hex_order_real_wind_b, hex_order_img_wind_b, n_of_neighbors_wind_b = np.zeros((params['n_windows'], 2)), np.zeros((params['n_windows'], 2)), np.zeros(params['n_windows'])
        
        for i in range(params['n_windows']):
            hex_order_real_wind_b[i, 0] = np.nanmean(hex_order_real_b[params['startFrames'][i]:params['endFrames'][i]])
            hex_order_real_wind_b[i, 1] = np.nanstd(hex_order_real_b[params['startFrames'][i]:params['endFrames'][i]]) * 1 / np.sqrt(len(params['blue_particle_idx']))
            hex_order_img_wind_b[i, 0] = np.nanmean(hex_order_img_b[params['startFrames'][i]:params['endFrames'][i]])
            hex_order_img_wind_b[i, 1] = np.nanstd(hex_order_img_b[params['startFrames'][i]:params['endFrames'][i]]) * 1 / np.sqrt(len(params['blue_particle_idx']))
            n_of_neighbors_wind_b[i] = np.mean(n_of_neighbors_b[params['startFrames'][i]:params['endFrames'][i]])

    if len(params['red_particle_idx']) > 0:
        velocity_pol_r = np.zeros(params['n_windows'])
        for step in tqdm(range(params['n_windows']), desc = "    Computing red droplets velocity polarization "):
            start = params['startFrames'][step]
            end = params['endFrames'][step]
            velocity_pol_r[step] = np.mean(np.linalg.norm(np.mean(orientations[start:end, params['red_mask']], axis = 1), axis = 1))

        res_red = compute_hex_order(positions[:, params['red_mask']], np.mean(radii[:, params['red_mask']], axis = 1), description = "    Computing red droplets hexatic order         ")
        hex_order_real_r, hex_order_img_r, n_of_neighbors_r, n_of_neighbors_r_std = res_red[:, 0], res_red[:, 1], res_red[:, 2], res_red[:, 3]
        hex_order_real_wind_r, hex_order_img_wind_r, n_of_neighbors_wind_r = np.zeros((params['n_windows'], 2)), np.zeros((params['n_windows'], 2)), np.zeros(params['n_windows'])

        for i in range(params['n_windows']):
            hex_order_real_wind_r[i, 0] = np.nanmean(hex_order_real_r[params['startFrames'][i]:params['endFrames'][i]])
            hex_order_real_wind_r[i, 1] = np.nanstd(hex_order_real_r[params['startFrames'][i]:params['endFrames'][i]]) * 1 / np.sqrt(len(params['red_particle_idx']))
            hex_order_img_wind_r[i, 0] = np.nanmean(hex_order_img_r[params['startFrames'][i]:params['endFrames'][i]])
            hex_order_img_wind_r[i, 1] = np.nanstd(hex_order_img_r[params['startFrames'][i]:params['endFrames'][i]]) * 1 / np.sqrt(len(params['red_particle_idx']))
            n_of_neighbors_wind_r[i] = np.mean(n_of_neighbors_r[params['startFrames'][i]:params['endFrames'][i]])
        
    if len(params['red_particle_idx']) > 0 and len(params['blue_particle_idx']) > 0:
        res_full = compute_hex_order(positions, np.mean(radii, axis = 1), description = "    Computing all droplets hexatic order         ")
        hex_order_real, hex_order_img, n_of_neighbors, n_of_neighbors_std = res_full[:, 0], res_full[:, 1], res_full[:, 2], res_full[:, 3]
        hex_order_real_wind, hex_order_img_wind, n_of_neighbors_wind = np.zeros((params['n_windows'], 2)), np.zeros((params['n_windows'], 2)), np.zeros(params['n_windows'])
        
        for i in range(params['n_windows']):
            hex_order_real_wind[i, 0] = np.nanmean(hex_order_real[params['startFrames'][i]:params['endFrames'][i]])
            hex_order_real_wind[i, 1] = np.nanstd(hex_order_real[params['startFrames'][i]:params['endFrames'][i]]) * 1 / np.sqrt(params['n_particles'])
            hex_order_img_wind[i, 0] = np.nanmean(hex_order_img[params['startFrames'][i]:params['endFrames'][i]])
            hex_order_img_wind[i, 1] = np.nanstd(hex_order_img[params['startFrames'][i]:params['endFrames'][i]]) * 1 / np.sqrt(params['n_particles'])
            n_of_neighbors_wind[i] = np.mean(n_of_neighbors[params['startFrames'][i]:params['endFrames'][i]])
            
    params['stages_colors'] = np.zeros(params['n_windows'])
    for i in range(params['n_windows']):
        for j in range(params['n_stages']):
            if params['startFrames'][i] >= params['frames_stages'][j]:
                params['stages_colors'][i] = j
    params['stages_colors'] = [params['stages_shades'][int(i)] for i in params['stages_colors']]

    if 1:
        fig, ax = plt.subplots(1, 1, figsize = (8, 4), sharex = True)
        if len(params['blue_particle_idx']) > 0:
            ax.plot(params['window_center_sec'], velocity_pol_b, "-b")
        if len(params['red_particle_idx']) > 0:
            ax.plot(params['window_center_sec'], velocity_pol_r, "-r")
        for i, frame in enumerate(params['frames_stages']):
            ax.bar(frame/params['fps'], height = 2000, width = params['window_length'], bottom = -10, color = params['stages_shades'][i], alpha = 0.5, label = f"Stage {i + 1}")
        ax.set(ylabel = r'$\Phi$', ylim = (0, 1), xlabel = 'Window Time [s]', title = f"Velocity polarization of system {params['system_name']}")
        ax.grid()
        ax.legend(loc = (0.1, 0.5), fontsize = 10)
        if params['video_selection'] in ['25b25r_lowconc_1', '25b25r_lowconc_2', '25b25r_lowconc_3', '25b25r_lowconc_5', '25b25r_lowconc_6']:
            ax.set(xlim = (-200, 14000))
        if save_plots:
            plt.savefig(f"./{params['res_path']}/order_analysis/velocity_polarization.png", bbox_inches="tight")
            plt.savefig(f"./{params['pdf_res_path']}/order_analysis/velocity_polarization.pdf", bbox_inches="tight")
        if show_plots:
            plt.show()
        else:
            plt.close()
            
        fig, (ax, ax1, ax2) = plt.subplots(3, 1, figsize = (10, 8), sharex=True)
        if len(params['blue_particle_idx']) > 0:
            ax.plot(params['window_center_sec'], hex_order_real_wind_b[:, 0], "b", label = "Blue")
            ax.fill_between(params['window_center_sec'], hex_order_real_wind_b[:, 0] - 2*hex_order_real_wind_b[:, 1], hex_order_real_wind_b[:, 0] + 2*hex_order_real_wind_b[:, 1], color = "b", alpha = 0.3)
            ax1.plot(params['window_center_sec'], hex_order_img_wind_b[:, 0], "b", label = "Blue")
            ax1.fill_between(params['window_center_sec'], hex_order_img_wind_b[:, 0] - 2*hex_order_img_wind_b[:, 1], hex_order_img_wind_b[:, 0] + 2*hex_order_img_wind_b[:, 1], color = "b", alpha = 0.3)
            ax2.plot(params['window_center_sec'], n_of_neighbors_wind_b, "b", label = "Blue")

        if len(params['red_particle_idx']) > 0:
            ax.plot(params['window_center_sec'], hex_order_real_wind_r[:, 0], "r", label = "Red")
            ax.fill_between(params['window_center_sec'], hex_order_real_wind_r[:, 0] - 2*hex_order_real_wind_r[:, 1], hex_order_real_wind_r[:, 0] + 2*hex_order_real_wind_r[:, 1], color = "r", alpha = 0.3)
            ax1.plot(params['window_center_sec'], hex_order_img_wind_r[:, 0], "r", label = "Red")
            ax1.fill_between(params['window_center_sec'], hex_order_img_wind_r[:, 0] - 2*hex_order_img_wind_r[:, 1], hex_order_img_wind_r[:, 0] + 2*hex_order_img_wind_r[:, 1], color = "r", alpha = 0.3)
            ax2.plot(params['window_center_sec'], n_of_neighbors_wind_r, "r", label = "Red")
            
        if len(params['red_particle_idx']) > 0 and len(params['blue_particle_idx']) > 0:
            ax.plot(params['window_center_sec'], hex_order_real_wind[:, 0], "k", label = "All droplets")
            ax.fill_between(params['window_center_sec'], hex_order_real_wind[:, 0] - 2*hex_order_real_wind[:, 1], hex_order_real_wind[:, 0] + 2*hex_order_real_wind[:, 1], color = "k", alpha = 0.3)
            ax1.plot(params['window_center_sec'], hex_order_img_wind[:, 0], "k", label = "All droplets")
            ax1.fill_between(params['window_center_sec'], hex_order_img_wind[:, 0] - 2*hex_order_img_wind[:, 1], hex_order_img_wind[:, 0] + 2*hex_order_img_wind[:, 1], color = "k", alpha = 0.3)
            ax2.plot(params['window_center_sec'], n_of_neighbors_wind, "k", label = "All droplets")

        ax.set(ylabel = 'Real part', ylim = (-0.1, 1.1), title = f"Hexatic order parameter -- {params['system_name']}")
        ax1.set( ylabel = 'Imaginary part', ylim = (-0.5, 0.5))
        ax2.set(xlabel = 'Window time [s]', ylabel = 'N of neighbors', ylim = (0, 6))
        for i, frame in enumerate(params['frames_stages']):
            ax.bar(frame/params['fps'], height = 2000, width = params['window_length'], bottom = -10, color = params['stages_shades'][i], alpha = 0.5, label = f"Stage {i + 1}")
            ax1.bar(frame/params['fps'], height = 2000, width = params['window_length'], bottom = -10, color = params['stages_shades'][i], alpha = 0.5, label = f"Stage {i + 1}")
            ax2.bar(frame/params['fps'], height = 2000, width = params['window_length'], bottom = -10, color = params['stages_shades'][i], alpha = 0.5, label = f"Stage {i + 1}")
        ax.grid(linewidth = 0.5)
        ax1.grid(linewidth = 0.5)
        ax2.grid(linewidth = 0.5)
        ax.legend()
        if params['video_selection'] in ['25b25r_lowconc_1', '25b25r_lowconc_2', '25b25r_lowconc_3', '25b25r_lowconc_5', '25b25r_lowconc_6']:
            ax2.set(xlim = (-200, 14000))
        plt.tight_layout()
        if save_plots:
            plt.savefig(f"{params['res_path']}/order_analysis/hex_order_windowed.png", bbox_inches="tight")
            plt.savefig(f"{params['pdf_res_path']}/order_analysis/hex_order_windowed.pdf", format='pdf', bbox_inches="tight")
        if show_plots:
            plt.show()
        else:
            plt.close()
        
        
        if len(params['blue_particle_idx']) > 0:
            test_pos = positions[:, ~params['red_mask']]
            radii_b = radii[:, ~params['red_mask']]

            fig, ax = plt.subplots(3, 3, figsize = (12, 12), sharex = 'row', sharey = 'row')
            frame = np.argmin(hex_order_real_b)
            n_of_neighbors_temp, theta = get_neighbours_props(test_pos[frame], np.mean(radii_b))
            ax[0, 0].imshow(get_frame(video, params['initial_offset'] + frame*params['subsample_factor'], params['xmin'], params['ymin'], params['xmax'], params['ymax'], params['resolution'], params['crop_verb']))
            for i in range(test_pos[frame].shape[0]):
                ax[0, 0].add_artist(plt.Circle((test_pos[frame][i, 0], test_pos[frame][i, 1]), radii_b[frame, i], color = 'blue', fill = True))
            ax[0, 0].set(title = f"Min -- {np.round(np.min(hex_order_real_b), 2)}", xlim = (0, params['resolution']), ylim = (0, params['resolution']), xticks = [], yticks = [])
            ax[1, 0].hist(n_of_neighbors_temp, bins = np.arange(0, 10), density = True, align = 'mid')
            ax[1, 0].set(xlabel = 'N', ylabel = 'N of neighbors pdf', xticks = np.arange(0, 10) + 0.5, xticklabels = np.arange(0, 10))
            ax[2, 0].hist(np.array(theta), bins = np.linspace(-np.pi, np.pi, 20), density = True, align = 'mid')
            ax[2, 0].set(xlabel = 'Angle [rad]', ylabel = 'Angle of neighbors pdf',  xticks = [-np.pi, -2*np.pi/3, -np.pi/3, -np.pi/6, 0, np.pi/6, np.pi/3, 2*np.pi/3, np.pi], xticklabels = [r'-$\pi$', r'-$\frac{2\pi}{3}$', r'-$\frac{\pi}{3}$', r'-$\frac{\pi}{6}$', '0', r'$\frac{\pi}{6}$', r'$\frac{\pi}{3}$', r'$\frac{2\pi}{3}$', r'$\pi$'])

            frame = np.argmax(hex_order_real_b)
            n_of_neighbors_temp, theta = get_neighbours_props(test_pos[frame], np.mean(radii_b))
            ax[0, 1].imshow(get_frame(video, params['initial_offset'] + frame*params['subsample_factor'], params['xmin'], params['ymin'], params['xmax'], params['ymax'], params['resolution'], params['crop_verb']))
            for i in range(test_pos[frame].shape[0]):
                ax[0, 1].add_artist(plt.Circle((test_pos[frame][i, 0], test_pos[frame][i, 1]), radii_b[frame, i], color = 'blue', fill = True))
                #ax[0, 1].add_artist(plt.Circle((test_pos[frame][i, 0], test_pos[frame][i, 1]), 2*1.5*np.mean(radii_b), color = 'blue', fill = False))
            ax[0, 1].set(title = f"Max -- {np.round(np.max(hex_order_real_b), 2)}", xlim = (0, params['resolution']), ylim = (0, params['resolution']), xticks = [], yticks = [])
            ax[1, 1].hist(n_of_neighbors_temp, bins = np.arange(0, 10), density = True, align = 'mid')
            ax[1, 1].set(xlabel = 'N', xticks = np.arange(0, 10) + 0.5, xticklabels = np.arange(0, 10))
            ax[2, 1].hist(np.array(theta), bins = np.linspace(-np.pi, np.pi, 20), density = True, align = 'mid')
            ax[2, 1].set(xlabel = 'Angle [rad]', xticks = [-np.pi, -2*np.pi/3, -np.pi/3, -np.pi/6, 0, np.pi/6, np.pi/3, 2*np.pi/3, np.pi], xticklabels = [r'-$\pi$', r'-$\frac{2\pi}{3}$', r'-$\frac{\pi}{3}$', r'-$\frac{\pi}{6}$', '0', r'$\frac{\pi}{6}$', r'$\frac{\pi}{3}$', r'$\frac{2\pi}{3}$', r'$\pi$'])

            frame = params['n_frames'] - 1
            n_of_neighbors_temp, theta = get_neighbours_props(test_pos[frame], np.mean(radii_b))
            ax[0, 2].imshow(get_frame(video, params['initial_offset'] + frame*params['subsample_factor'], params['xmin'], params['ymin'], params['xmax'], params['ymax'], params['resolution'], params['crop_verb']))
            for i in range(test_pos[frame].shape[0]):
                ax[0, 2].add_artist(plt.Circle((test_pos[frame][i, 0], test_pos[frame][i, 1]), radii_b[frame, i], color = 'blue', fill = True))
                #ax[0, 2].add_artist(plt.Circle((test_pos[frame][i, 0], test_pos[frame][i, 1]), 2*1.5*np.mean(radii_b), color = 'blue', fill = False))
            ax[0, 2].set(title = f'Last frame -- {np.round(hex_order_real_b[frame], 2)}', xlim = (0, params['resolution']), ylim = (0, params['resolution']), xticks = [], yticks = [])
            ax[1, 2].hist(n_of_neighbors_temp, bins = np.arange(0, 10), density = True, align = 'mid')
            ax[1, 2].set(xlabel = 'N', xticks = np.arange(0, 10) + 0.5, xticklabels = np.arange(0, 10))
            ax[2, 2].hist(np.array(theta), bins = np.linspace(-np.pi, np.pi, 20), density = True, align = 'mid')
            ax[2, 2].set(xlabel = 'Angle [rad]', xticks = [-np.pi, -2*np.pi/3, -np.pi/3, -np.pi/6, 0, np.pi/6, np.pi/3, 2*np.pi/3, np.pi], xticklabels = [r'-$\pi$', r'-$\frac{2\pi}{3}$', r'-$\frac{\pi}{3}$', r'-$\frac{\pi}{6}$', '0', r'$\frac{\pi}{6}$', r'$\frac{\pi}{3}$', r'$\frac{2\pi}{3}$', r'$\pi$'])
            plt.suptitle(f"Examples of droplet configurations -- {params['system_name']}")
            plt.tight_layout()
            if save_plots:
                plt.savefig(f"./{params['res_path']}/order_analysis/hex_order_examples_blue.png", bbox_inches="tight")
                plt.savefig(f"./{params['pdf_res_path']}/order_analysis/hex_order_examples_blue.pdf", bbox_inches="tight")
            if show_plots:
                plt.show()
            else:
                plt.close()
        
        if len(params['red_particle_idx']) > 0:
            test_pos = positions[:, params['red_mask']]
            radii_r = radii[:, params['red_mask']]

            fig, ax = plt.subplots(3, 3, figsize = (12, 12), sharex = 'row', sharey = 'row')
            frame = np.argmin(hex_order_real_r)
            n_of_neighbors_temp, theta = get_neighbours_props(test_pos[frame], np.mean(radii_r))
            ax[0, 0].imshow(get_frame(video, params['initial_offset'] + frame*params['subsample_factor'], params['xmin'], params['ymin'], params['xmax'], params['ymax'], params['resolution'], params['crop_verb']))
            for i in range(test_pos[frame].shape[0]):
                ax[0, 0].add_artist(plt.Circle((test_pos[frame][i, 0], test_pos[frame][i, 1]), radii_r[frame, i], color = 'red', fill = True))
            ax[0, 0].set(title = f'Min -- {np.round(np.min(hex_order_real_r), 2)}', xlim = (0, params['resolution']), ylim = (0, params['resolution']), xticks = [], yticks = [])
            ax[1, 0].hist(n_of_neighbors_temp, bins = np.arange(0, 10), density = True, align = 'mid')
            ax[1, 0].set(xlabel = 'N', ylabel = 'N of neighbors pdf', xticks = np.arange(0, 10) + 0.5, xticklabels = np.arange(0, 10))
            ax[2, 0].hist(np.array(theta), bins = np.linspace(-np.pi, np.pi, 20), density = True, align = 'mid')
            ax[2, 0].set(xlabel = 'Angle [rad]', ylabel = 'Angle of neighbors pdf',  xticks = [-np.pi, -2*np.pi/3, -np.pi/3, -np.pi/6, 0, np.pi/6, np.pi/3, 2*np.pi/3, np.pi], xticklabels = [r'-$\pi$', r'-$\frac{2\pi}{3}$', r'-$\frac{\pi}{3}$', r'-$\frac{\pi}{6}$', '0', r'$\frac{\pi}{6}$', r'$\frac{\pi}{3}$', r'$\frac{2\pi}{3}$', r'$\pi$'])

            frame = np.argmax(hex_order_real_r)
            n_of_neighbors_temp, theta = get_neighbours_props(test_pos[frame], np.mean(radii_r))
            ax[0, 1].imshow(get_frame(video, params['initial_offset'] + frame*params['subsample_factor'], params['xmin'], params['ymin'], params['xmax'], params['ymax'], params['resolution'], params['crop_verb']))
            for i in range(test_pos[frame].shape[0]):
                ax[0, 1].add_artist(plt.Circle((test_pos[frame][i, 0], test_pos[frame][i, 1]), radii_r[frame, i], color = 'red', fill = True))
            ax[0, 1].set(title = f'Max -- {np.round(np.max(hex_order_real_r), 2)}', xlim = (0, params['resolution']), ylim = (0, params['resolution']), xticks = [], yticks = [])
            ax[1, 1].hist(n_of_neighbors_temp, bins = np.arange(0, 10), density = True, align = 'mid')
            ax[1, 1].set(xlabel = 'N', xticks = np.arange(0, 10) + 0.5, xticklabels = np.arange(0, 10))
            ax[2, 1].hist(np.array(theta), bins = np.linspace(-np.pi, np.pi, 20), density = True, align = 'mid')
            ax[2, 1].set(xlabel = 'Angle [rad]', xticks = [-np.pi, -2*np.pi/3, -np.pi/3, -np.pi/6, 0, np.pi/6, np.pi/3, 2*np.pi/3, np.pi], xticklabels = [r'-$\pi$', r'-$\frac{2\pi}{3}$', r'-$\frac{\pi}{3}$', r'-$\frac{\pi}{6}$', '0', r'$\frac{\pi}{6}$', r'$\frac{\pi}{3}$', r'$\frac{2\pi}{3}$', r'$\pi$'])

            frame = params['n_frames'] - 1
            n_of_neighbors_temp, theta = get_neighbours_props(test_pos[frame], np.mean(radii_r))
            ax[0, 2].imshow(get_frame(video, params['initial_offset'] + frame*params['subsample_factor'], params['xmin'], params['ymin'], params['xmax'], params['ymax'], params['resolution'], params['crop_verb']))
            for i in range(test_pos[frame].shape[0]):
                ax[0, 2].add_artist(plt.Circle((test_pos[frame][i, 0], test_pos[frame][i, 1]), radii_r[frame, i], color = 'red', fill = True))
            ax[0, 2].set(title = f'Last frame -- {np.round(hex_order_real_r[frame], 2)}', xlim = (0, params['resolution']), ylim = (0, params['resolution']), xticks = [], yticks = [])
            ax[1, 2].hist(n_of_neighbors_temp, bins = np.arange(0, 10), density = True, align = 'mid')
            ax[1, 2].set(xlabel = 'N', xticks = np.arange(0, 10) + 0.5, xticklabels = np.arange(0, 10))
            ax[2, 2].hist(np.array(theta), bins = np.linspace(-np.pi, np.pi, 20), density = True, align = 'mid')
            ax[2, 2].set(xlabel = 'Angle [rad]', xticks = [-np.pi, -2*np.pi/3, -np.pi/3, -np.pi/6, 0, np.pi/6, np.pi/3, 2*np.pi/3, np.pi], xticklabels = [r'-$\pi$', r'-$\frac{2\pi}{3}$', r'-$\frac{\pi}{3}$', r'-$\frac{\pi}{6}$', '0', r'$\frac{\pi}{6}$', r'$\frac{\pi}{3}$', r'$\frac{2\pi}{3}$', r'$\pi$'])
            plt.suptitle(f"Examples of droplet configurations -- {params['system_name']}")
            plt.tight_layout()
            if save_plots:
                plt.savefig(f"./{params['res_path']}/order_analysis/hex_order_examples_red.png", bbox_inches="tight")
                plt.savefig(f"./{params['pdf_res_path']}/order_analysis/hex_order_examples_red.pdf", bbox_inches="tight")
            if show_plots:
                plt.show()
            else:
                plt.close()
            
        if len(params['red_particle_idx']) > 0 and len(params['blue_particle_idx']) > 0:
            fig, ax = plt.subplots(3, 3, figsize = (12, 12), sharex = 'row', sharey = 'row')
            frame = np.argmin(hex_order_real)
            n_of_neighbors_temp, theta = get_neighbours_props(positions[frame], np.mean(radii))
            ax[0, 0].imshow(get_frame(video, params['initial_offset'] + frame*params['subsample_factor'], params['xmin'], params['ymin'], params['xmax'], params['ymax'], params['resolution'], params['crop_verb']))
            for i in range(positions[frame].shape[0]):
                ax[0, 0].add_artist(plt.Circle((positions[frame][i, 0], positions[frame][i, 1]), radii[frame, i], color = 'black', fill = True))
            ax[0, 0].set(title = f'Min -- {np.round(np.min(hex_order_real), 2)}', xlim = (0, params['resolution']), ylim = (0, params['resolution']), xticks = [], yticks = [])
            ax[1, 0].hist(n_of_neighbors_temp, bins = np.arange(0, 10), density = True, align = 'mid')
            ax[1, 0].set(xlabel = 'N', ylabel = 'N of neighbors pdf', xticks = np.arange(0, 10) + 0.5, xticklabels = np.arange(0, 10))
            ax[2, 0].hist(np.array(theta), bins = np.linspace(-np.pi, np.pi, 20), density = True, align = 'mid')
            ax[2, 0].set(xlabel = 'Angle [rad]', ylabel = 'Angle of neighbors pdf',  xticks = [-np.pi, -2*np.pi/3, -np.pi/3, -np.pi/6, 0, np.pi/6, np.pi/3, 2*np.pi/3, np.pi], xticklabels = [r'-$\pi$', r'-$\frac{2\pi}{3}$', r'-$\frac{\pi}{3}$', r'-$\frac{\pi}{6}$', '0', r'$\frac{\pi}{6}$', r'$\frac{\pi}{3}$', r'$\frac{2\pi}{3}$', r'$\pi$'])

            frame = np.argmax(hex_order_real)
            n_of_neighbors_temp, theta = get_neighbours_props(positions[frame], np.mean(radii))
            ax[0, 1].imshow(get_frame(video, params['initial_offset'] + frame*params['subsample_factor'], params['xmin'], params['ymin'], params['xmax'], params['ymax'], params['resolution'], params['crop_verb']))
            for i in range(positions[frame].shape[0]):
                ax[0, 1].add_artist(plt.Circle((positions[frame][i, 0], positions[frame][i, 1]), radii[frame, i], color = 'black', fill = True))
            ax[0, 1].set(title = f'Max -- {np.round(np.max(hex_order_real), 2)}', xlim = (0, params['resolution']), ylim = (0, params['resolution']), xticks = [], yticks = [])
            ax[1, 1].hist(n_of_neighbors_temp, bins = np.arange(0, 10), density = True, align = 'mid')
            ax[1, 1].set(xlabel = 'N', xticks = np.arange(0, 10) + 0.5, xticklabels = np.arange(0, 10))
            ax[2, 1].hist(np.array(theta), bins = np.linspace(-np.pi, np.pi, 20), density = True, align = 'mid')
            ax[2, 1].set(xlabel = 'Angle [rad]', xticks = [-np.pi, -2*np.pi/3, -np.pi/3, -np.pi/6, 0, np.pi/6, np.pi/3, 2*np.pi/3, np.pi], xticklabels = [r'-$\pi$', r'-$\frac{2\pi}{3}$', r'-$\frac{\pi}{3}$', r'-$\frac{\pi}{6}$', '0', r'$\frac{\pi}{6}$', r'$\frac{\pi}{3}$', r'$\frac{2\pi}{3}$', r'$\pi$'])

            frame = params['n_frames'] - 1
            n_of_neighbors_temp, theta = get_neighbours_props(positions[frame], np.mean(radii))
            ax[0, 2].imshow(get_frame(video, params['initial_offset'] + frame*params['subsample_factor'], params['xmin'], params['ymin'], params['xmax'], params['ymax'], params['resolution'], params['crop_verb']))
            for i in range(positions[frame].shape[0]):
                ax[0, 2].add_artist(plt.Circle((positions[frame][i, 0], positions[frame][i, 1]), radii[frame, i], color = 'black', fill = True))
            ax[0, 2].set(title = f'Last frame -- {np.round(hex_order_real[frame], 2)}', xlim = (0, params['resolution']), ylim = (0, params['resolution']), xticks = [], yticks = [])
            ax[1, 2].hist(n_of_neighbors_temp, bins = np.arange(0, 10), density = True, align = 'mid')
            ax[1, 2].set(xlabel = 'N', xticks = np.arange(0, 10) + 0.5, xticklabels = np.arange(0, 10))
            ax[2, 2].hist(np.array(theta), bins = np.linspace(-np.pi, np.pi, 20), density = True, align = 'mid')
            ax[2, 2].set(xlabel = 'Angle [rad]', xticks = [-np.pi, -2*np.pi/3, -np.pi/3, -np.pi/6, 0, np.pi/6, np.pi/3, 2*np.pi/3, np.pi], xticklabels = [r'-$\pi$', r'-$\frac{2\pi}{3}$', r'-$\frac{\pi}{3}$', r'-$\frac{\pi}{6}$', '0', r'$\frac{\pi}{6}$', r'$\frac{\pi}{3}$', r'$\frac{2\pi}{3}$', r'$\pi$'])
            plt.suptitle(f"Examples of droplet configurations -- {params['system_name']}")
            plt.tight_layout()
            if save_plots:
                plt.savefig(f"./{params['res_path']}/order_analysis/hex_order_examples_full.png", bbox_inches="tight")
                plt.savefig(f"./{params['pdf_res_path']}/order_analysis/hex_order_examples_full.pdf", bbox_inches="tight")
            if show_plots:
                plt.show()
            else:
                plt.close()
            
        fig, (ax, ax1) = plt.subplots(2, 1, figsize = (8, 6), sharex = True)
        if len(params['blue_particle_idx']) > 0:
            ax.plot(params['window_center_sec'], velocity_pol_b, '-b')
            ax1.plot(params['window_center_sec'], hex_order_real_wind_b[:, 0], "b", label = "Blue")
            ax1.fill_between(params['window_center_sec'], hex_order_real_wind_b[:, 0] - 2*hex_order_real_wind_b[:, 1], hex_order_real_wind_b[:, 0] + 2*hex_order_real_wind_b[:, 1], color = "b", alpha = 0.3)

        if len(params['red_particle_idx']) > 0:
            ax.plot(params['window_center_sec'], velocity_pol_r, '-r')
            ax1.plot(params['window_center_sec'], hex_order_real_wind_r[:, 0], "r", label = "Red")
            ax1.fill_between(params['window_center_sec'], hex_order_real_wind_r[:, 0] - 2*hex_order_real_wind_r[:, 1], hex_order_real_wind_r[:, 0] + 2*hex_order_real_wind_r[:, 1], color = "r", alpha = 0.3)
        
        if len(params['red_particle_idx']) > 0 and len(params['blue_particle_idx']) > 0:
            ax1.plot(params['window_center_sec'], hex_order_real_wind[:, 0], "k", label = "All")
            ax1.fill_between(params['window_center_sec'], hex_order_real_wind[:, 0] - 2*hex_order_real_wind[:, 1], hex_order_real_wind[:, 0] + 2*hex_order_real_wind[:, 1], color = "k", alpha = 0.3)
    
        for i, frame in enumerate(params['frames_stages']):
            ax.bar(frame/params['fps'], height = 2000, width = params['window_length'], bottom = -10, color = params['stages_shades'][i], alpha = 0.5, label = f"Stage {i + 1}")
            ax1.bar(frame/params['fps'], height = 2000, width = params['window_length'], bottom = -10, color = params['stages_shades'][i], alpha = 0.5)
            
        ax.set(ylabel = r'$\Phi$', ylim = (0, 1), title = f"Dynamical and structural order of system {params['system_name']}")
        ax1.set(ylabel = r'$\langle Re(\phi_6) \rangle$', xlabel = 'Window time [s]', ylim = (-0.1, 1.1))
        ax.grid()
        ax1.grid()
        ax.legend(loc = (0.1, 0.5), fontsize = 10)
        ax1.legend(loc = (0.1, 0.5), fontsize = 10)
        if params['video_selection'] in ['25b25r_lowconc_1', '25b25r_lowconc_2', '25b25r_lowconc_3', '25b25r_lowconc_5', '25b25r_lowconc_6']:
            ax.set(xlim = (-200, 14000))
        plt.tight_layout()
        if save_plots:
            plt.savefig(f"./{params['res_path']}/order_analysis/polarization_and_hexatic.png", bbox_inches="tight")
            plt.savefig(f"./{params['pdf_res_path']}/order_analysis/polarization_and_hexatic.pdf", bbox_inches="tight")
        if show_plots:
            plt.show()
        else:
            plt.close()
            
    if (len(params['blue_particle_idx']) > 0) & (len(params['red_particle_idx']) > 0):
        return (velocity_pol_b, velocity_pol_r), (hex_order_real_wind_b, hex_order_real_wind_r, hex_order_real_wind)
    elif (len(params['blue_particle_idx']) > 0) & (len(params['red_particle_idx']) == 0):
        return (velocity_pol_b, None), (hex_order_real_wind_b, None, None)
    elif (len(params['blue_particle_idx']) == 0) & (len(params['red_particle_idx']) > 0):
        return (None, velocity_pol_r), (None, hex_order_real_wind_r, None)
    