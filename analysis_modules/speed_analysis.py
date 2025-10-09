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

from analysis_utils import MB_2D, MB_2D_generalized, fit_hist, get_trajs, speed_windowed, onClick


def run_speed_analysis(trajectories, params, show_plots, save_plots, run_analysis_verb, animated_plot_results):
    
    print("    Global speed distribution analysis...")
    
    if len(params['blue_particle_idx']) > 0:
        blueTrajs = get_trajs(trajectories.loc[trajectories.particle.isin(params['blue_particle_idx'])], params['fps'], params['pxDimension'])
        v_blue = ys.speed_ensemble(blueTrajs, step = 1)
        mean_v_blue = np.mean(v_blue)
        speed_bins = np.linspace(0, (np.max(v_blue) + 1)/params['pxDimension'], 1000)*params['pxDimension']

    if len(params['red_particle_idx']) > 0:
        redTrajs = get_trajs(trajectories.loc[trajectories.particle.isin(params['red_particle_idx'])], params['fps'], params['pxDimension'])
        v_red = ys.speed_ensemble(redTrajs, step = 1)
        mean_v_red = np.mean(v_red)
        speed_bins = np.linspace(0, (np.max(v_red) + 1)/params['pxDimension'], 1000)*params['pxDimension']

    if len(params['red_particle_idx']) > 0 & len(params['blue_particle_idx']) > 0:
        speed_bins = np.linspace(0, (np.max([np.max(v_blue), np.max(v_red)]) + 1)/params['pxDimension'], 1000)*params['pxDimension']

    speed_bin_centers = (speed_bins[1:] + speed_bins[:-1]) / 2
    x_interval_for_fit = np.linspace(speed_bins[0], speed_bins[-1], 10000)
    print('    Windowed speed distribution analysis...')
    if len(params['blue_particle_idx']) > 0:
        if run_analysis_verb:
            mean_speed_b, std_speed_b, speed_distr_b, fit_results_wind_b, r2_wind_b, fit_results_wind_g_b, r2_g_wind_b = speed_windowed(params['n_windows'], params['startFrames'], params['endFrames'],
                                                                                                                                        trajectories.loc[trajectories.particle.isin(params['blue_particle_idx'])], params['fps'], params['pxDimension'], 
                                                                                                                                        speed_bins, speed_bin_centers, 
                                                                                                                                        progress_verb=True, description = '    Computing windowed speed distribution for blue droplets')
            if os.path.isfile(f"./{params['analysis_data_path']}/speed_analysis/speed_distr_b.npz"):
                os.remove(f"./{params['analysis_data_path']}/speed_analysis/speed_distr_b.npz")
            np.savez(f"./{params['analysis_data_path']}/speed_analysis/speed_distr_b.npz",  mean_speed_b = mean_speed_b, std_speed_b = std_speed_b, speed_distr_b = speed_distr_b, fit_results_wind_b = fit_results_wind_b, r2_wind_b = r2_wind_b, fit_results_wind_g_b = fit_results_wind_g_b, r2_g_wind_b = r2_g_wind_b)
        else:
            mean_speed_b, std_speed_b, speed_distr_b, fit_results_wind_b, r2_wind_b, fit_results_wind_g_b, r2_g_wind_b = np.load(f"./{params['analysis_data_path']}/speed_analysis/speed_distr_b.npz").values()

    if len(params['red_particle_idx']) > 0:
        if run_analysis_verb:
            mean_speed_r, std_speed_r, speed_distr_r, fit_results_wind_r, r2_wind_r, fit_results_wind_g_r, r2_g_wind_r = speed_windowed(params['n_windows'], params['startFrames'], params['endFrames'],
                                                                                                                                        trajectories.loc[trajectories.particle.isin(params['red_particle_idx'])], params['fps'], params['pxDimension'],
                                                                                                                                        speed_bins, speed_bin_centers,
                                                                                                                                        progress_verb=True, description = '    Computing windowed speed distribution for red droplets ')
            if os.path.isfile(f"./{params['analysis_data_path']}/speed_analysis/speed_distr_r.npz"):
                os.remove(f"./{params['analysis_data_path']}/speed_analysis/speed_distr_r.npz")
            np.savez(f"./{params['analysis_data_path']}/speed_analysis/speed_distr_r.npz",  mean_speed_r = mean_speed_r, std_speed_r = std_speed_r, speed_distr_r = speed_distr_r, fit_results_wind_r = fit_results_wind_r, r2_wind_r = r2_wind_r, fit_results_wind_g_r = fit_results_wind_g_r, r2_g_wind_r = r2_g_wind_r)
        else:
            mean_speed_r, std_speed_r, speed_distr_r, fit_results_wind_r, r2_wind_r, fit_results_wind_g_r, r2_g_wind_r = np.load(f"./{params['analysis_data_path']}/speed_analysis/speed_distr_r.npz").values()
            
            
    if 1:
        gs = gridspec.GridSpec(2, 10)
        fig = plt.figure(figsize = (18, 6))
        i, step = 0, params['steps_plot'][0]
        ax1 = fig.add_subplot(gs[0, :2])
        if len(params['blue_particle_idx']) > 0:
            ax1.bar(speed_bin_centers, speed_distr_b[step], width = speed_bins[1] - speed_bins[0], color = 'b', alpha = 0.5)
        if len(params['red_particle_idx']) > 0:
            ax1.bar(speed_bin_centers, speed_distr_r[step], width = speed_bins[1] - speed_bins[0], color = 'r', alpha = 0.5)
        ax1.set(title = f"Stage {i + 1}", ylim = (0, 8), xlim = (-.1, 2))
        ax1.set(xlabel = f"v [{params['speed_units']}]", ylabel = 'pdf [s/mm]')
        ax1.grid(linewidth = 0.2)
        i, step = 1, params['steps_plot'][1]
        ax2 = fig.add_subplot(gs[0, 2:4], sharex = ax1, sharey = ax1)
        if len(params['blue_particle_idx']) > 0:
            ax2.bar(speed_bin_centers, speed_distr_b[step], width = speed_bins[1] - speed_bins[0], color = 'b', alpha = 0.5)
        if len(params['red_particle_idx']) > 0:
            ax2.bar(speed_bin_centers, speed_distr_r[step], width = speed_bins[1] - speed_bins[0], color = 'r', alpha = 0.5)
        ax2.set(title = f"Stage {i + 1}", ylim = (0, 8), xlim = (-.1, 2))
        ax2.set(xlabel = f"v [{params['speed_units']}]")
        ax2.grid(linewidth = 0.2)
        plt.setp(ax2.get_yticklabels(), visible=False)
        i, step = 2, params['steps_plot'][2]
        ax3 = fig.add_subplot(gs[0, 4:6], sharex = ax1, sharey = ax1)
        if len(params['blue_particle_idx']) > 0:
            ax3.bar(speed_bin_centers, speed_distr_b[step], width = speed_bins[1] - speed_bins[0], color = 'b', alpha = 0.5)
        if len(params['red_particle_idx']) > 0:
            ax3.bar(speed_bin_centers, speed_distr_r[step], width = speed_bins[1] - speed_bins[0], color = 'r', alpha = 0.5)
        ax3.set(title = f"Stage {i + 1}", ylim = (0, 8), xlim = (-.1, 2))
        ax3.set(xlabel = f"v [{params['speed_units']}]")
        ax3.grid(linewidth = 0.2)
        plt.setp(ax3.get_yticklabels(), visible=False)
        i, step = 3, params['steps_plot'][3]
        ax4 = fig.add_subplot(gs[0, 6:8], sharex = ax1, sharey = ax1)
        if len(params['blue_particle_idx']) > 0:
            ax4.bar(speed_bin_centers, speed_distr_b[step], width = speed_bins[1] - speed_bins[0], color = 'b', alpha = 0.5)
        if len(params['red_particle_idx']) > 0:
            ax4.bar(speed_bin_centers, speed_distr_r[step], width = speed_bins[1] - speed_bins[0], color = 'r', alpha = 0.5)
        ax4.set(title = f"Stage {i + 1}", ylim = (0, 8), xlim = (-.1, 2))
        ax4.set(xlabel = f"v [{params['speed_units']}]")
        ax4.grid(linewidth = 0.2)
        plt.setp(ax4.get_yticklabels(), visible=False)
        i, step = 4, params['steps_plot'][4]
        ax5 = fig.add_subplot(gs[0, 8:10], sharex = ax1, sharey = ax1)
        if len(params['blue_particle_idx']) > 0:
            ax5.bar(speed_bin_centers, speed_distr_b[step], width = speed_bins[1] - speed_bins[0], color = 'b', alpha = 0.5)
        if len(params['red_particle_idx']) > 0:
            ax5.bar(speed_bin_centers, speed_distr_r[step], width = speed_bins[1] - speed_bins[0], color = 'r', alpha = 0.5)
        ax5.set(title = f"Stage {i + 1}", ylim = (0, 8), xlim = (-.1, 2)) 
        ax5.set(xlabel = f"v [{params['speed_units']}]")
        ax5.grid(linewidth = 0.2)
        #ax5.legend(fontsize = 10)
        plt.setp(ax5.get_yticklabels(), visible=False)
        ax6 = fig.add_subplot(gs[1, :5])
        if len(params['blue_particle_idx']) > 0:
            ax6.plot(params['window_center_sec'], mean_speed_b, 'b-',label = 'Blue droplets')
        if len(params['red_particle_idx']) > 0:
            ax6.plot(params['window_center_sec'], mean_speed_r, 'r-', label = 'Red droplets')
        for i, frame in enumerate(params['frames_stages']):
            ax6.bar(frame/params['fps'], height = 2000, width = params['window_length'], bottom = -10, color = params['stages_shades'][i], alpha = 0.5)
        ax6.set(ylim = (-0.1, 2.5), xlim = (-200, params['max_window_sec']))
        ax6.set(ylabel = r'$\langle v \rangle$ [mm/s]', title = 'Mean speed')
        ax6.legend(loc = (0.1, 0.7), fontsize = 10)
        ax6.grid(linewidth = 0.2)
        
        ax7 = fig.add_subplot(gs[1, 5:])
        if len(params['blue_particle_idx']) > 0:
            ax7.plot(params['window_center_sec'], std_speed_b, 'b-',label = 'Blue droplets')
        if len(params['red_particle_idx']) > 0:
            ax7.plot(params['window_center_sec'], std_speed_r, 'r-', label = 'Red droplets')
        for i, frame in enumerate(params['frames_stages']):
            ax7.bar(frame/params['fps'], height = 2000, width = params['window_length'], bottom = -10, color = params['stages_shades'][i], alpha = 0.5)
        ax7.grid(linewidth = 0.2)
        ax7.legend(fontsize = 10)
        ax7.set_title('Speed std')
        ax7.set(ylim = (-0.1, 2), xlim = (-200, params['max_window_sec']))
        
        ax7.set(ylabel = 'std(v) [mm/s]')
        ax7.legend(loc = (0.1, 0.7), fontsize = 10)
        ax1.text(0.0, 1.0, 'a)', transform=(ax1.transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)), fontsize='medium', va='bottom')
        ax2.text(0.0, 1.0, 'b)', transform=(ax2.transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)), fontsize='medium', va='bottom')
        ax3.text(0.0, 1.0, 'c)', transform=(ax3.transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)), fontsize='medium', va='bottom')
        ax4.text(0.0, 1.0, 'd)', transform=(ax4.transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)), fontsize='medium', va='bottom')
        ax5.text(0.0, 1.0, 'e)', transform=(ax5.transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)), fontsize='medium', va='bottom')
        ax6.text(0.0, 1.0, 'f)', transform=(ax6.transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)), fontsize='medium', va='bottom')
        ax7.text(0.0, 1.0, 'g)', transform=(ax7.transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)), fontsize='medium', va='bottom')
        plt.tight_layout()
        if save_plots:
            plt.savefig(f"./{params['res_path']}/speed_analysis/speed_wind_stages_{params['n_stages']}.png", bbox_inches='tight')
            plt.savefig(f"./{params['pdf_res_path']}/speed_analysis/speed_wind_stages_{params['n_stages']}.pdf", bbox_inches='tight')
        if show_plots:
            plt.show()
        else:
            plt.close()


        if animated_plot_results:
            fig, (ax, ax1) = plt.subplots(1, 2, figsize = (10, 4), sharex = True, sharey = True)
            anim_running = True

            def update_plot(step):
                # update titles 
                title.set_text(f"Speed distribution of system {params['system_name']} at  " + r'$t_w$' + f" = {params['startFrames'][step]/params['fps'] + params['window_length']/2} s")
                
                if len(params['blue_particle_idx']) > 0: 
                    line_b.set_ydata(MB_2D(x_interval_for_fit, *fit_results_wind_b[step, :, 0]))
                    line_b1.set_ydata(MB_2D_generalized(x_interval_for_fit, *fit_results_wind_g_b[step, :, 0]))
                    for i, b in enumerate(bar_container_b):
                        b.set_height(speed_distr_b[step, i])

                if len(params['red_particle_idx']) > 0: 
                    line_r.set_ydata(MB_2D(x_interval_for_fit, *fit_results_wind_r[step, :, 0]))
                    line_r1.set_ydata(MB_2D_generalized(x_interval_for_fit, *fit_results_wind_g_r[step, :, 0]))
                    for i, b in enumerate(bar_container_r):
                        b.set_height(speed_distr_r[step, i])

                if (len(params['blue_particle_idx']) > 0) & (len(params['red_particle_idx']) == 0):
                    return bar_container_b, line_b, line_b1
                if (len(params['blue_particle_idx']) == 0) & (len(params['red_particle_idx']) > 0):
                    return bar_container_r, line_r, line_r1
                if (len(params['blue_particle_idx']) > 0) & (len(params['red_particle_idx']) > 0):
                    return bar_container_b, bar_container_r, line_b, line_b1, line_r, line_r1

            title = plt.suptitle(f"Speed distribution of system {params['system_name']} at  " + r'$t_w$' + f" = {params['startFrames'][0]/params['fps'] + params['window_length']/2} s")

            if len(params['blue_particle_idx']) > 0:
                bar_container_b = ax.bar(speed_bin_centers, speed_distr_b[0], width = speed_bins[1] - speed_bins[0], color = 'b', alpha = 0.5)
                line_b, = ax.plot(x_interval_for_fit, MB_2D(x_interval_for_fit, *fit_results_wind_b[0, :, 0]), label = '2D MB fit')
                line_b1, = ax.plot(x_interval_for_fit, MB_2D_generalized(x_interval_for_fit, *fit_results_wind_g_b[0, :, 0]), label = 'Generalized 2D MB fit')
            
            if len(params['red_particle_idx']) > 0:
                bar_container_r = ax1.bar(speed_bin_centers, speed_distr_r[0], width = speed_bins[1] - speed_bins[0], color = 'r', alpha = 0.5)
                line_r, = ax1.plot(x_interval_for_fit, MB_2D(x_interval_for_fit, *fit_results_wind_r[0, :, 0]), label = '2D MB fit')
                line_r1, = ax1.plot(x_interval_for_fit, MB_2D_generalized(x_interval_for_fit, *fit_results_wind_g_r[0, :, 0]), label = 'Generalized 2D MB fit')

            ax.set(xlabel = f"v [{params['speed_units']}]", ylabel = 'pdf [s/mm]')
            ax.grid(linewidth = 0.2)
            ax1.grid(linewidth = 0.2)
            ax.legend(loc = (0.1, 0.6), fontsize = 10)
            ax1.set(xlabel = f"v [{params['speed_units']}]", xlim = (-.1, 5), ylim = (0, 8))
            plt.tight_layout()
            fig.canvas.mpl_connect('button_press_event', onClick)
            ani = FuncAnimation(fig, update_plot, params['n_windows'], repeat=True, blit=False)
            writer = FFMpegWriter(fps = 10, metadata = dict(artist='skandiz'), extra_args=['-vcodec', 'libx264'])
            if save_plots: ani.save(f"./{params['res_path']}/speed_analysis/speed_wind.mp4", writer = writer, dpi = 300)
            if show_plots: 
                plt.show()
            else:
                plt.close()
                
    if (len(params['blue_particle_idx']) > 0) & (len(params['red_particle_idx']) > 0):
        return (mean_speed_b, mean_speed_r)
    elif (len(params['blue_particle_idx']) > 0) & (len(params['red_particle_idx']) == 0):
        return (mean_speed_b, None)
    elif (len(params['blue_particle_idx']) == 0) & (len(params['red_particle_idx']) > 0):
        return (None, mean_speed_r)


    
    