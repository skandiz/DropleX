import os
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({
	'font.size': 12,               # General font size
	'axes.titlesize': 14,          # Title font size
	'axes.labelsize': 12,          # Axis label font size
	'legend.fontsize': 10,         # Legend font size
	'xtick.labelsize': 10,
	'ytick.labelsize': 10})
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.transforms import ScaledTranslation
import yupi.stats as ys
from tqdm import tqdm
from yupi import Trajectory, WindowType, DiffMethod

from analysis_utils import normal_distr, wrapped_lorentzian_distr, fit_hist, turning_angles_windowed, get_trajs, onClick

def run_turning_analysis(trajectories, frames, params, show_plots, save_plots, run_analysis_verb, animated_plot_results):    
    turn_angles_bins = np.linspace(-np.pi, np.pi, 601)
    turn_angles_bin_centers = turn_angles_bins[:-1] + np.diff(turn_angles_bins) / 2
    x_interval_for_fit_turn = np.linspace(turn_angles_bins[0], turn_angles_bins[-1], 10000)
    
    print('    Windowed turning angles analysis...')
    if len(params['blue_particle_idx']) > 0:
        if run_analysis_verb:
            turn_angles_b, gaussian_fit_results_wind_b, gaussian_r2_wind_b, lorentzian_fit_results_wind_b, lorentzian_r2_wind_b = turning_angles_windowed(params['n_windows'], params['startFrames'], params['endFrames'], trajectories.loc[trajectories.particle.isin(params['blue_particle_idx'])], params['fps'], params['pxDimension'],
                                                                                                                                                          turn_angles_bins, turn_angles_bin_centers, progress_verb = True, description = '    Computing windowed turning angles for blue droplets')
            if os.path.isfile(f"./{params['analysis_data_path']}/turning_angles_analysis/turning_angles_windowed_blue.npz"):
                os.remove(f"./{params['analysis_data_path']}/turning_angles_analysis/turning_angles_windowed_blue.npz")
            np.savez(f"./{params['analysis_data_path']}/turning_angles_analysis/turning_angles_windowed_blue.npz", turn_angles_b = turn_angles_b, gaussian_fit_results_wind_b = gaussian_fit_results_wind_b, gaussian_r2_wind_b = gaussian_r2_wind_b, lorentzian_fit_results_wind_b = lorentzian_fit_results_wind_b, lorentzian_r2_wind_b = lorentzian_r2_wind_b)
        else:
            data = np.load(f"./{params['analysis_data_path']}/turning_angles_analysis/turning_angles_windowed_blue.npz")
            turn_angles_b = data['turn_angles_b']
            gaussian_fit_results_wind_b = data['gaussian_fit_results_wind_b']
            gaussian_r2_wind_b = data['gaussian_r2_wind_b']
            lorentzian_fit_results_wind_b = data['lorentzian_fit_results_wind_b']
            lorentzian_r2_wind_b = data['lorentzian_r2_wind_b']
            
    
    if len(params['red_particle_idx']) > 0:
        if run_analysis_verb:
            turn_angles_r, gaussian_fit_results_wind_r, gaussian_r2_wind_r, lorentzian_fit_results_wind_r, lorentzian_r2_wind_r = turning_angles_windowed(params['n_windows'], params['startFrames'], params['endFrames'], trajectories.loc[trajectories.particle.isin(params['red_particle_idx'])], params['fps'], params['pxDimension'],
                                                                                                                                                          turn_angles_bins, turn_angles_bin_centers, progress_verb = True, description = '    Computing windowed turning angles for red droplets ')
            if os.path.isfile(f"./{params['analysis_data_path']}/turning_angles_analysis/turning_angles_windowed_red.npz"):
                os.remove(f"./{params['analysis_data_path']}/turning_angles_analysis/turning_angles_windowed_red.npz")
            np.savez(f"./{params['analysis_data_path']}/turning_angles_analysis/turning_angles_windowed_red.npz", turn_angles_r = turn_angles_r, gaussian_fit_results_wind_r = gaussian_fit_results_wind_r, gaussian_r2_wind_r = gaussian_r2_wind_r, lorentzian_fit_results_wind_r = lorentzian_fit_results_wind_r, lorentzian_r2_wind_r = lorentzian_r2_wind_r)
        else:
            data = np.load(f"./{params['analysis_data_path']}/turning_angles_analysis/turning_angles_windowed_red.npz")
            turn_angles_r = data['turn_angles_r']
            gaussian_fit_results_wind_r = data['gaussian_fit_results_wind_r']
            gaussian_r2_wind_r = data['gaussian_r2_wind_r']
            lorentzian_fit_results_wind_r = data['lorentzian_fit_results_wind_r']
            lorentzian_r2_wind_r = data['lorentzian_r2_wind_r']

    if 1:
        fig, ax = plt.subplots(1, 1, figsize = (10, 4))
        if len(params['blue_particle_idx']) > 0:
            ax.plot(params['window_center_sec'], lorentzian_r2_wind_b, 'b', label = 'Lorentzian Fit')
            ax.plot(params['window_center_sec'], gaussian_r2_wind_b, 'b--', label = 'Gaussian Fit')

        if len(params['red_particle_idx']) > 0:
            ax.plot(params['window_center_sec'], gaussian_r2_wind_r, 'r--', label = 'Gaussian Fit')
            ax.plot(params['window_center_sec'], lorentzian_r2_wind_r, 'r', label = 'Lorentzian fit') 
        
        for i, frame in enumerate(params['frames_stages']):
            ax.bar(frame/params['fps'], 20000, params['window_length'], bottom = -100, color = params['stages_shades'][i], alpha = 0.5, label = f"Stage {i+1}")
        
        ax.grid(linewidth = 0.2)
        ax.legend(fontsize = 10)
        
        ax.set(ylim = (0.85, 1), xlabel = r'$t_w$ [s]', ylabel = r'$R^2$', title = f"R² comparison fit of the turning angles distribution of system {params['system_name']}")
        if save_plots: 
            plt.savefig(f"./{params['res_path']}/turning_angles_analysis/r2_comparison.png", bbox_inches='tight')
            plt.savefig(f"./{params['pdf_res_path']}/turning_angles_analysis/r2_comparison.pdf", bbox_inches='tight')
        if show_plots:
            plt.show()
        else:
            plt.close()
        
        gs = gridspec.GridSpec(2, 10)
        fig = plt.figure(figsize = (18, 6))
        i, step = 0, params['steps_plot'][0]
        ax1 = fig.add_subplot(gs[0, :2])
        if len(params['blue_particle_idx']) > 0:
            ax1.bar(turn_angles_bin_centers, turn_angles_b[step], width = np.diff(turn_angles_bins)[0], color = 'b', alpha = 0.5, label = 'Blue droplets')
            ax1.plot(x_interval_for_fit_turn, wrapped_lorentzian_distr(x_interval_for_fit_turn, *lorentzian_fit_results_wind_b[step, :, 0]), color = 'blue', label = 'Wrapped Lorentzian fit')
        if len(params['red_particle_idx']) > 0:
            ax1.bar(turn_angles_bin_centers, turn_angles_r[step], width = np.diff(turn_angles_bins)[0], color = 'r', alpha = 0.5, label = 'Red droplets')
            ax1.plot(x_interval_for_fit_turn, wrapped_lorentzian_distr(x_interval_for_fit_turn, *lorentzian_fit_results_wind_r[step, :, 0]), color = 'red', label = 'Wrapped Lorentzian fit')
        ax1.grid(linewidth = 0.2)
        ax1.set_xticks([-np.pi, -np.pi/2, -np.pi/4, -np.pi/8, 0, np.pi/8, np.pi/4, np.pi/2, np.pi], [r'-$\pi$', r'$-\pi/2$', r'$-\pi/4$', r'$-\pi/8$', '$0$', r'$\pi/8$', r'$\pi/4$', r'$\pi/2$', r'$\pi$'])
        ax1.set(title = f"Stage {i + 1}", xlabel = r'$\Delta \theta$ [rad]', ylabel = 'pdf [1/rad]', xlim = (-np.pi/4, np.pi/4))
        ax1.set(ylim = (0, 6))
        i, step = 1, params['steps_plot'][1]
        ax2 = fig.add_subplot(gs[0, 2:4], sharex = ax1, sharey = ax1)
        if len(params['blue_particle_idx']) > 0:
            ax2.bar(turn_angles_bin_centers, turn_angles_b[step], width = np.diff(turn_angles_bins)[0], color = 'b', alpha = 0.5, label = 'Blue droplets')
            ax2.plot(x_interval_for_fit_turn, wrapped_lorentzian_distr(x_interval_for_fit_turn, *lorentzian_fit_results_wind_b[step, :, 0]), color = 'blue', label = 'Wrapped Lorentzian fit')
        if len(params['red_particle_idx']) > 0:
            ax2.bar(turn_angles_bin_centers, turn_angles_r[step], width = np.diff(turn_angles_bins)[0], color = 'r', alpha = 0.5, label = 'Red droplets')
            ax2.plot(x_interval_for_fit_turn, wrapped_lorentzian_distr(x_interval_for_fit_turn, *lorentzian_fit_results_wind_r[step, :, 0]), color = 'red', label = 'Wrapped Lorentzian fit')
        ax2.grid(linewidth = 0.2)
        ax2.set(title = f"Stage {i + 1}", xlabel = r'$\Delta \theta$ [rad]')
        plt.setp(ax2.get_yticklabels(), visible=False)
        i, step = 2, params['steps_plot'][2]
        ax3 = fig.add_subplot(gs[0, 4:6], sharex = ax1, sharey = ax1)
        if len(params['blue_particle_idx']) > 0:
            ax3.bar(turn_angles_bin_centers, turn_angles_b[step], width = np.diff(turn_angles_bins)[0], color = 'b', alpha = 0.5, label = 'Blue droplets')
            ax3.plot(x_interval_for_fit_turn, wrapped_lorentzian_distr(x_interval_for_fit_turn, *lorentzian_fit_results_wind_b[step, :, 0]), color = 'blue', label = 'Wrapped Lorentzian fit')
        if len(params['red_particle_idx']) > 0:
            ax3.bar(turn_angles_bin_centers, turn_angles_r[step], width = np.diff(turn_angles_bins)[0], color = 'r', alpha = 0.5, label = 'Red droplets')
            ax3.plot(x_interval_for_fit_turn, wrapped_lorentzian_distr(x_interval_for_fit_turn, *lorentzian_fit_results_wind_r[step, :, 0]), color = 'red', label = 'Wrapped Lorentzian fit')
        ax3.grid(linewidth = 0.2)
        ax3.set(title = f"Stage {i + 1}", xlabel = r'$\Delta \theta$ [rad]')
        plt.setp(ax3.get_yticklabels(), visible=False)
        i, step = 3, params['steps_plot'][3]
        ax4 = fig.add_subplot(gs[0, 6:8], sharex = ax1, sharey = ax1)
        if len(params['blue_particle_idx']) > 0:
            ax4.bar(turn_angles_bin_centers, turn_angles_b[step], width = np.diff(turn_angles_bins)[0], color = 'b', alpha = 0.5, label = 'Blue droplets')
            ax4.plot(x_interval_for_fit_turn, wrapped_lorentzian_distr(x_interval_for_fit_turn, *lorentzian_fit_results_wind_b[step, :, 0]), color = 'blue', label = 'Wrapped Lorentzian fit')
        if len(params['red_particle_idx']) > 0:
            ax4.bar(turn_angles_bin_centers, turn_angles_r[step], width = np.diff(turn_angles_bins)[0], color = 'r', alpha = 0.5, label = 'Red droplets')
            ax4.plot(x_interval_for_fit_turn, wrapped_lorentzian_distr(x_interval_for_fit_turn, *lorentzian_fit_results_wind_r[step, :, 0]), color = 'red', label = 'Wrapped Lorentzian fit')
        ax4.grid(linewidth = 0.2)
        ax4.set(title = f"Stage {i + 1}", xlabel = r'$\Delta \theta$ [rad]')
        plt.setp(ax4.get_yticklabels(), visible=False)
        i, step = 4, params['steps_plot'][4]
        ax5 = fig.add_subplot(gs[0, 8:10], sharex = ax1, sharey = ax1)
        if len(params['blue_particle_idx']) > 0:
            ax5.bar(turn_angles_bin_centers, turn_angles_b[step], width = np.diff(turn_angles_bins)[0], color = 'b', alpha = 0.5, label = 'Blue droplets')
            ax5.plot(x_interval_for_fit_turn, wrapped_lorentzian_distr(x_interval_for_fit_turn, *lorentzian_fit_results_wind_b[step, :, 0]), color = 'blue', label = 'Wrapped Lorentzian fit')
        if len(params['red_particle_idx']) > 0:
            ax5.bar(turn_angles_bin_centers, turn_angles_r[step], width = np.diff(turn_angles_bins)[0], color = 'r', alpha = 0.5, label = 'Red droplets')
            ax5.plot(x_interval_for_fit_turn, wrapped_lorentzian_distr(x_interval_for_fit_turn, *lorentzian_fit_results_wind_r[step, :, 0]), color = 'red', label = 'Wrapped Lorentzian fit')
        ax5.grid(linewidth = 0.2)
        ax5.set(title = f"Stage {i + 1}", xlabel = r'$\Delta \theta$ [rad]')
        ax5.legend(fontsize = 10)
        plt.setp(ax5.get_yticklabels(), visible=False)
        ax6 = fig.add_subplot(gs[1, :5])
        if len(params['blue_particle_idx']) > 0:
            ax6.plot(params['window_center_sec'], lorentzian_fit_results_wind_b[:, 0, 0], 'b')
        if len(params['red_particle_idx']) > 0:
            ax6.plot(params['window_center_sec'], lorentzian_fit_results_wind_r[:, 0, 0], 'r')
            
        for i, frame in enumerate(params['frames_stages']):
            ax6.bar(frame/params['fps'], 20000, params['window_length'], bottom = -100, color = params['stages_shades'][i], alpha = 0.5, label = f"Stage {i+1}")
        ax6.set(ylabel = r'$\gamma \; [rad]$', xlabel = r'$t_w$ [s]')#, title = 'Scale factor')
        if params['system_name'].startswith('25b25r'):
            if params['subsample_factor'] == 3:
                ax6.set(ylim = (0, 1), xlim = (-200, params['max_window_sec']))
            else:
                ax6.set(ylim = (0, 0.2), xlim = (-200, params['max_window_sec']))
        elif params['system_name'].startswith('1b'):
            if params['subsample_factor'] == 3:
                ax6.set(ylim = (0, 1), xlim = (-200, params['max_window_sec']))
            else:
                ax6.set(ylim = (0, 0.12), xlim = (-200, params['max_window_sec']))
        else:
            ax6.set(ylim = (0, 1))
        ax6.grid(linewidth = 0.2)
        ax6.legend(['Blue droplets', 'Red droplets'], fontsize = 10, loc = (0.09, 0.7))
        ax7 = fig.add_subplot(gs[1, 5:])
        if len(params['blue_particle_idx']) > 0:
            ax7.plot(params['window_center_sec'], lorentzian_fit_results_wind_b[:, 1, 0], 'b')
        if len(params['red_particle_idx']) > 0:
            ax7.plot(params['window_center_sec'], lorentzian_fit_results_wind_r[:, 1, 0], 'r')
            
        for i, frame in enumerate(params['frames_stages']):
            ax7.bar(frame/params['fps'], 20000, params['window_length'], bottom = -100, color = params['stages_shades'][i], alpha = 0.5, label = f"Stage {i+1}")
        ax7.set(ylabel = r'$\mu \; [rad]$', xlabel = r'$t_w$ [s]')#, title = 'Mean')
        ax7.set(ylim = (-0.01, 0.01), xlim = (-200, params['max_window_sec']))
        
        #ax7.legend(loc = (0.1, 0.4), fontsize = 10)
        ax7.grid(linewidth = 0.2)
        ax1.text(0.0, 1.0, 'a)', transform=(ax1.transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)), fontsize='medium', va='bottom')
        ax2.text(0.0, 1.0, 'b)', transform=(ax2.transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)), fontsize='medium', va='bottom')
        ax3.text(0.0, 1.0, 'c)', transform=(ax3.transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)), fontsize='medium', va='bottom')
        ax4.text(0.0, 1.0, 'd)', transform=(ax4.transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)), fontsize='medium', va='bottom')
        ax5.text(0.0, 1.0, 'e)', transform=(ax5.transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)), fontsize='medium', va='bottom')
        ax6.text(0.0, 1.0, 'f)', transform=(ax6.transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)), fontsize='medium', va='bottom')
        ax7.text(0.0, 1.0, 'g)', transform=(ax7.transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)), fontsize='medium', va='bottom')
        #plt.suptitle(f"Turning angles distribution of system {params['system_name']}")
        plt.tight_layout()
        if save_plots:
            plt.savefig(f"./{params['res_path']}/turning_angles_analysis/turning_angles_wind_stages_{params['n_stages']}.png", bbox_inches='tight')
            plt.savefig(f"./{params['pdf_res_path']}/turning_angles_analysis/turning_angles_wind_stages_{params['n_stages']}.pdf", bbox_inches='tight')
        if show_plots:
            plt.show()
        else:
            plt.close()

        if animated_plot_results:
            fig, (ax, ax1) = plt.subplots(2, 1, figsize=(8, 5), sharex = True, sharey = True)
            anim_running = True
            def update_plot(frame):
                title.set_text(f"Turning angles distribution of system {params['system_name']} at  " + r'$t_w$' + f"= {params['startFrames'][frame]/params['fps'] + params['window_length']/2} s")
                if len(params['blue_particle_idx']) > 0:
                    line_b.set_ydata(normal_distr(x_interval_for_fit_turn, *gaussian_fit_results_wind_b[frame, :, 0]))
                    line_b1.set_ydata(wrapped_lorentzian_distr(x_interval_for_fit_turn, *lorentzian_fit_results_wind_b[frame, :, 0]))
                    for i, b in enumerate(bar_container_b):
                        b.set_height(turn_angles_b[frame, i])
        
                if len(params['red_particle_idx']) > 0:
                    line_r.set_ydata(normal_distr(x_interval_for_fit_turn, *gaussian_fit_results_wind_r[frame, :, 0]))
                    line_r1.set_ydata(wrapped_lorentzian_distr(x_interval_for_fit_turn, *lorentzian_fit_results_wind_r[frame, :, 0]))
                    for i, b in enumerate(bar_container_r):
                        b.set_height(turn_angles_r[frame, i])
        
                if (len(params['blue_particle_idx']) > 0) & (len(params['red_particle_idx']) == 0):
                    return bar_container_b, line_b, line_b1
                
                elif (len(params['blue_particle_idx']) == 0) & (len(params['red_particle_idx']) > 0):
                    return bar_container_r, line_r, line_r1
                
                elif (len(params['blue_particle_idx']) > 0) & (len(params['red_particle_idx']) > 0):
                    return bar_container_b, bar_container_r, line_b, line_r, line_b1, line_r1
            
            title = ax.set_title(f"Turning angles distribution of system {params['system_name']} at  " + r'$t_w$' + f"= {params['startFrames'][0]/params['fps'] + params['window_length']/2} s")

            if len(params['blue_particle_idx']) > 0:
                line_b, = ax.plot(x_interval_for_fit_turn, normal_distr(x_interval_for_fit_turn, *gaussian_fit_results_wind_b[0, :, 0]), label = 'Gaussian fit')
                line_b1, = ax.plot(x_interval_for_fit_turn, wrapped_lorentzian_distr(x_interval_for_fit_turn, *lorentzian_fit_results_wind_b[0, :, 0]), label = 'Lorentzian fit')
                bar_container_b = ax.bar(turn_angles_bin_centers, turn_angles_b[0], width = np.diff(turn_angles_bins)[0], color = 'b', alpha = 0.5, label = 'Blue droplets')
            
            if len(params['red_particle_idx']) > 0:
                line_r, = ax1.plot(x_interval_for_fit_turn, normal_distr(x_interval_for_fit_turn, *gaussian_fit_results_wind_r[0, :, 0]), label = 'Gaussian fit')
                line_r1, = ax1.plot(x_interval_for_fit_turn, wrapped_lorentzian_distr(x_interval_for_fit_turn, *lorentzian_fit_results_wind_r[0, :, 0]), label = 'Lorentzian fit')
                bar_container_r = ax1.bar(turn_angles_bin_centers, turn_angles_r[0], width = np.diff(turn_angles_bins)[0], color = 'r', alpha = 0.5, label = 'Red droplets')
            
            ax.set(ylabel = 'pdf', ylim = (0, 20))
            ax1.set(ylabel = 'pdf', ylim = (0, 20))
            ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], [r'-$\pi$', r'$-\frac{\pi}{2}$', '$0$', r'$\frac{\pi}{2}$', r'$\pi$'])
            ax1.set_xticks([-np.pi, -np.pi/2, -np.pi/4, -np.pi/8, 0, np.pi/8, np.pi/4, np.pi/2, np.pi], [r'-$\pi$', r'$-\frac{\pi}{2}$', r'$-\frac{\pi}{4}$', r'$-\frac{\pi}{8}$', '$0$', r'$\frac{\pi}{8}$', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$', r'$\pi$'])
            
            ax1.set(xlim = (-np.pi/4, np.pi/4))
            
            ax.grid(linewidth = 0.2)
            ax1.grid(linewidth = 0.2)
            ax.legend(fontsize = 10)
            ax1.legend(fontsize = 10)
            
            plt.tight_layout()
            fig.canvas.mpl_connect('button_press_event', onClick)
            ani = FuncAnimation(fig, update_plot, params['n_windows'], blit=False)
            writer = FFMpegWriter(fps = 10, metadata = dict(artist='skandiz'), extra_args=['-vcodec', 'libx264'])
            ani.save(f"./{params['res_path']}/turning_angles_analysis/turn_ang_wind.mp4", writer = writer, dpi = 300)
            if show_plots:
                plt.show()
            else:
                plt.close()
                
    
    
    if (len(params['blue_particle_idx']) > 0) & (len(params['red_particle_idx']) > 0):
        return (turn_angles_b, turn_angles_r), (lorentzian_fit_results_wind_b, lorentzian_fit_results_wind_r), (gaussian_fit_results_wind_b, gaussian_fit_results_wind_r), (turn_angles_bins, turn_angles_bin_centers, x_interval_for_fit_turn)
    elif (len(params['blue_particle_idx']) > 0) & (len(params['red_particle_idx']) == 0):
        return (turn_angles_b, None), (lorentzian_fit_results_wind_b, None), (gaussian_fit_results_wind_b, None), (turn_angles_bins, turn_angles_bin_centers, x_interval_for_fit_turn)
    elif (len(params['blue_particle_idx']) == 0) & (len(params['red_particle_idx']) > 0):
        return (None, turn_angles_r), (None, lorentzian_fit_results_wind_r), (None, gaussian_fit_results_wind_r), (turn_angles_bins, turn_angles_bin_centers, x_interval_for_fit_turn)
