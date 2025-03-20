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

    if len(params['blue_particle_idx']) > 0:
        blue_speed_distr, _ = np.histogram(v_blue, speed_bins, density = True)

        # fit speed distribution with 2D Maxwell Boltzmann distribution
        fit_results_gaussian_b, r2_blue = fit_hist(blue_speed_distr, speed_bin_centers, MB_2D, np.array([1.]), maxfev_ = 25000)
        print(f"    MB fit results             -- Blue droplets --> R² = {np.round(r2_blue, 2)} -- μ = {np.round(mean_v_blue, 3)} mm/s -- σ = {np.round(fit_results_gaussian_b[0, 0], 3)} ± {np.round(fit_results_gaussian_b[0, 1], 3)} mm/s")
        
        # fit speed distribution with a generalization of a 2D Maxwell Boltzmann distribution
        fit_results_gaussian_b_g, r2_blue_g  = fit_hist(blue_speed_distr, speed_bin_centers, MB_2D_generalized, [1., 2., 1.], maxfev_ = 25000)
        print(f"    Generalized MB fit results -- Blue droplets --> R² = {np.round(r2_blue_g, 3)} -- σ = {np.round(fit_results_gaussian_b_g[0, 0], 3)} ± {np.round(fit_results_gaussian_b_g[0, 1], 3)} mm/s -- b = {np.round(fit_results_gaussian_b_g[1, 0], 3)} ± {np.round(fit_results_gaussian_b_g[1, 1], 3)} -- A = {np.round(fit_results_gaussian_b_g[2, 0], 3)} ± {np.round(fit_results_gaussian_b_g[2, 1], 3)} ")

    if len(params['red_particle_idx']) > 0:
        red_speed_distr, _ = np.histogram(v_red, speed_bins, density = True)
        # fit speed distribution with 2D Maxwell Boltzmann distribution
        fit_results_gaussian_r, r2_red  = fit_hist(red_speed_distr, speed_bin_centers, MB_2D, np.array([1.]), maxfev_ = 25000)
        print(f"    MB fit results             -- Red droplets  --> R² = {np.round(r2_red,2)} -- μ = {np.round(mean_v_red, 3)} mm/s --σ = {np.round(fit_results_gaussian_r[0, 0], 3)} ± {np.round(fit_results_gaussian_r[0, 1], 3)} mm/s")
        
        # fit speed distribution with a generalization of a 2D Maxwell Boltzmann distribution
        fit_results_gaussian_r_g, r2_red_g = fit_hist(red_speed_distr, speed_bin_centers, MB_2D_generalized, [1., 2., 1.], maxfev_ = 25000)
        print(f"    Generalized MB fit results -- Red droplets  --> R² = {np.round(r2_red_g, 3)} -- σ = {np.round(fit_results_gaussian_r_g[0, 0], 4)} ± {np.round(fit_results_gaussian_r_g[0, 1], 4)} mm/s -- b = {np.round(fit_results_gaussian_r_g[1, 0], 3)} ± {np.round(fit_results_gaussian_r_g[1, 1], 3)} -- A = {np.round(fit_results_gaussian_r_g[2, 0], 3)} ± {np.round(fit_results_gaussian_r_g[2, 1], 3)}")
        
        
    if 1:    
        fig, (ax, ax1) = plt.subplots(1, 2, figsize = (12, 4), sharey=True, sharex = True)
        if len(params['blue_particle_idx']) > 0:
            ax.hist(v_blue, bins = speed_bins, **params['default_kwargs_blue'][0], label = 'Blue droplets')
            ax.plot(x_interval_for_fit, MB_2D(x_interval_for_fit, fit_results_gaussian_b[:, 0]), label = 'MB fit')
            ax.plot(x_interval_for_fit, MB_2D_generalized(x_interval_for_fit, *fit_results_gaussian_b_g[:, 0]), label = 'Generalized MB fit')
        ax.set(xlabel = f"v [{params['speed_units']}]", ylabel = 'pdf [s/mm]')
        ax.legend(fontsize = 10)
        ax.grid(linewidth = 0.2)
        if len(params['red_particle_idx']) > 0:
            ax1.hist(v_red, bins = speed_bins, **params['default_kwargs_red'][0], label = 'Red droplets')
            ax1.plot(x_interval_for_fit, MB_2D(x_interval_for_fit, fit_results_gaussian_r[:, 0]), label = 'MB fit')
            ax1.plot(x_interval_for_fit, MB_2D_generalized(x_interval_for_fit, *fit_results_gaussian_r_g[:, 0]), label = 'Generalized MB fit')
        if params['video_selection'] not in ['1b_&_1r_1', '1b_&_1r_2', '1b_&_1r_3']:
            ax1.set(xlabel = f"v [{params['speed_units']}]", xlim = (-.1, 5), ylim = (0, 7))
        ax1.legend(fontsize = 10)
        ax1.grid(linewidth = 0.2)
        ax.text(0.0, 1.0, 'a)', transform=(ax.transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)), fontsize='medium', va='bottom')
        ax1.text(0.0, 1.0, 'b)', transform=(ax1.transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)), fontsize='medium', va='bottom')
        plt.suptitle(f"Speed distribution of system {params['system_name']}")
        plt.tight_layout()
        if save_plots: 
            plt.savefig(f"./{params['res_path']}/speed_analysis/speed_distribution.png", bbox_inches='tight')
            plt.savefig(f"./{params['pdf_res_path']}/speed_analysis/speed_distribution.pdf", bbox_inches='tight')
        if show_plots: 
            plt.show()
        else:
            plt.close()
    
    print('    Windowed speed distribution analysis...')
    if len(params['blue_particle_idx']) > 0:
        if run_analysis_verb:
            mean_speed_b, std_speed_b, speed_distr_b, fit_results_wind_b, r2_wind_b, fit_results_wind_g_b, r2_g_wind_b = speed_windowed(params['n_windows'], params['startFrames'], params['endFrames'],
                                                                                                                                        trajectories, params['fps'], params['pxDimension'], 
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
                                                                                                                                        trajectories, params['fps'], params['pxDimension'],
                                                                                                                                        speed_bins, speed_bin_centers,
                                                                                                                                        progress_verb=True, description = '    Computing windowed speed distribution for red droplets ')
            if os.path.isfile(f"./{params['analysis_data_path']}/speed_analysis/speed_distr_r.npz"):
                os.remove(f"./{params['analysis_data_path']}/speed_analysis/speed_distr_r.npz")
            np.savez(f"./{params['analysis_data_path']}/speed_analysis/speed_distr_r.npz",  mean_speed_r = mean_speed_r, std_speed_r = std_speed_r, speed_distr_r = speed_distr_r, fit_results_wind_r = fit_results_wind_r, r2_wind_r = r2_wind_r, fit_results_wind_g_r = fit_results_wind_g_r, r2_g_wind_r = r2_g_wind_r)
        else:
            mean_speed_r, std_speed_r, speed_distr_r, fit_results_wind_r, r2_wind_r, fit_results_wind_g_r, r2_g_wind_r = np.load(f"./{params['analysis_data_path']}/speed_analysis/speed_distr_r.npz").values()
            
            
    if 1:
        gs = gridspec.GridSpec(2, 10)
        fig = plt.figure(figsize = (18, 8))
        i, step = 0, params['steps_plot'][0]
        ax1 = fig.add_subplot(gs[0, :2])
        if len(params['blue_particle_idx']) > 0:
            ax1.bar(speed_bin_centers, speed_distr_b[step], width = speed_bins[1] - speed_bins[0], color = 'b', alpha = 0.5)
            #ax1.plot(x, kde_blue[i], color='blue')
        if len(params['red_particle_idx']) > 0:
            ax1.bar(speed_bin_centers, speed_distr_r[step], width = speed_bins[1] - speed_bins[0], color = 'r', alpha = 0.5)
            #ax1.plot(x, kde_red[i], color='red')
        if params['video_selection'] in ['1b_&_1r_1', '1b_&_1r_2', '1b_&_1r_3']:
            ax1.set(title = f"Stage {i + 1}")
        else:
            ax1.set(title = f"Stage {i + 1}", ylim = (0, 8), xlim = (-.1, 2))
        ax1.set(xlabel = f"v [{params['speed_units']}]", ylabel = 'pdf [s/mm]')
        ax1.grid(linewidth = 0.2)
        i, step = 1, params['steps_plot'][1]
        ax2 = fig.add_subplot(gs[0, 2:4], sharex = ax1, sharey = ax1)
        if len(params['blue_particle_idx']) > 0:
            ax2.bar(speed_bin_centers, speed_distr_b[step], width = speed_bins[1] - speed_bins[0], color = 'b', alpha = 0.5)
            #ax2.plot(x, kde_blue[i], color='blue')
        if len(params['red_particle_idx']) > 0:
            ax2.bar(speed_bin_centers, speed_distr_r[step], width = speed_bins[1] - speed_bins[0], color = 'r', alpha = 0.5)
            #ax2.plot(x, kde_red[i], color='red')
        if params['video_selection'] in ['1b_&_1r_1', '1b_&_1r_2', '1b_&_1r_3']:
            ax2.set(title = f"Stage {i + 1}")
        else:
            ax2.set(title = f"Stage {i + 1}", ylim = (0, 8), xlim = (-.1, 2))
        ax2.set(xlabel = f"v [{params['speed_units']}]")
        ax2.grid(linewidth = 0.2)
        plt.setp(ax2.get_yticklabels(), visible=False)
        i, step = 2, params['steps_plot'][2]
        ax3 = fig.add_subplot(gs[0, 4:6], sharex = ax1, sharey = ax1)
        if len(params['blue_particle_idx']) > 0:
            ax3.bar(speed_bin_centers, speed_distr_b[step], width = speed_bins[1] - speed_bins[0], color = 'b', alpha = 0.5)
            #ax3.plot(x, kde_blue[i], color='blue')
        if len(params['red_particle_idx']) > 0:
            ax3.bar(speed_bin_centers, speed_distr_r[step], width = speed_bins[1] - speed_bins[0], color = 'r', alpha = 0.5)
            #ax3.plot(x, kde_red[i], color='red')
        if params['video_selection'] in ['1b_&_1r_1', '1b_&_1r_2', '1b_&_1r_3']:
            ax3.set(title = f"Stage {i + 1}")
        else:
            ax3.set(title = f"Stage {i + 1}", ylim = (0, 8), xlim = (-.1, 2))
        ax3.set(xlabel = f"v [{params['speed_units']}]")
        ax3.grid(linewidth = 0.2)
        plt.setp(ax3.get_yticklabels(), visible=False)
        i, step = 3, params['steps_plot'][3]
        ax4 = fig.add_subplot(gs[0, 6:8], sharex = ax1, sharey = ax1)
        if len(params['blue_particle_idx']) > 0:
            ax4.bar(speed_bin_centers, speed_distr_b[step], width = speed_bins[1] - speed_bins[0], color = 'b', alpha = 0.5)
            #ax4.plot(x, kde_blue[i], color='blue')
        if len(params['red_particle_idx']) > 0:
            ax4.bar(speed_bin_centers, speed_distr_r[step], width = speed_bins[1] - speed_bins[0], color = 'r', alpha = 0.5)
            #ax4.plot(x, kde_red[i], color='red')
        if params['video_selection'] in ['1b_&_1r_1', '1b_&_1r_2', '1b_&_1r_3']:
            ax4.set(title = f"Stage {i + 1}")
        else:
            ax4.set(title = f"Stage {i + 1}", ylim = (0, 8), xlim = (-.1, 2))
        ax4.set(xlabel = f"v [{params['speed_units']}]")
        ax4.grid(linewidth = 0.2)
        plt.setp(ax4.get_yticklabels(), visible=False)
        i, step = 4, params['steps_plot'][4]
        ax5 = fig.add_subplot(gs[0, 8:10], sharex = ax1, sharey = ax1)
        if len(params['blue_particle_idx']) > 0:
            ax5.bar(speed_bin_centers, speed_distr_b[step], width = speed_bins[1] - speed_bins[0], color = 'b', alpha = 0.5)
            #ax5.plot(x, kde_blue[i], color='blue')
        if len(params['red_particle_idx']) > 0:
            ax5.bar(speed_bin_centers, speed_distr_r[step], width = speed_bins[1] - speed_bins[0], color = 'r', alpha = 0.5)
            #ax5.plot(x, kde_red[i], color='red')
        if params['video_selection'] in ['1b_&_1r_1', '1b_&_1r_2', '1b_&_1r_3']:
            ax5.set(title = f"Stage {i + 1}")
        else:
            ax5.set(title = f"Stage {i + 1}", ylim = (0, 8), xlim = (-.1, 2)) 
        ax5.set(xlabel = f"v [{params['speed_units']}]")
        ax5.grid(linewidth = 0.2)
        ax5.legend(fontsize = 10)
        plt.setp(ax5.get_yticklabels(), visible=False)
        ax6 = fig.add_subplot(gs[1, :5])
        if len(params['blue_particle_idx']) > 0:
            ax6.plot(params['window_center_sec'], mean_speed_b, 'b-',label = 'Blue droplets')
        if len(params['red_particle_idx']) > 0:
            ax6.plot(params['window_center_sec'], mean_speed_r, 'r-', label = 'Red droplets')
        for i, frame in enumerate(params['frames_stages']):
            ax6.bar(frame/params['fps'], height = 2000, width = params['window_length'], bottom = -10, color = params['stages_shades'][i], alpha = 0.5)
        if params['video_selection'] in ['25b25r_lowconc_1', '25b25r_lowconc_2', '25b25r_lowconc_3', '25b25r_lowconc_5', '25b25r_lowconc_6']:
            ax6.set(ylim = (-0.1, 2.5), xlim = (-200, 14000))
        elif params['video_selection'] in ['25b25r-1', '25b25r-2']:
            ax6.set(ylim = (-0.1, 4))
        elif params['video_selection'] in ['1b_&_1r_1', '1b_&_1r_2', '1b_&_1r_3']:
            ax6.set(ylim = (-0.1, 40))
        else:
            ax6.set(ylim = (-0.1, 2))
        ax6.grid(linewidth = 0.2)
        ax6.set(ylabel = r'$\langle v \rangle$ [mm/s]', title = 'Mean speed')
        ax6.legend(loc = (0.1, 0.7), fontsize = 10)
        
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
        if params['video_selection'] in ['25b25r_lowconc_1', '25b25r_lowconc_2', '25b25r_lowconc_3', '25b25r_lowconc_5', '25b25r_lowconc_6']:
            ax7.set(ylim = (-0.1, 2), xlim = (-200, 14000))
        elif params['video_selection'] in ['25b25r-1', '25b25r-2']:
            ax7.set(ylim = (-0.1, 4))
        elif params['video_selection'] in ['1b_&_1r_1', '1b_&_1r_2', '1b_&_1r_3']:
            ax7.set(ylim = (-0.1, 15))
        else:
            ax7.set(ylim = (-0.1, 2))
        ax7.set(ylabel = 'std(v) [mm/s]')
        ax7.legend(loc = (0.1, 0.7), fontsize = 10)
        ax1.text(0.0, 1.0, 'a)', transform=(ax1.transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)), fontsize='medium', va='bottom')
        ax2.text(0.0, 1.0, 'b)', transform=(ax2.transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)), fontsize='medium', va='bottom')
        ax3.text(0.0, 1.0, 'c)', transform=(ax3.transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)), fontsize='medium', va='bottom')
        ax4.text(0.0, 1.0, 'd)', transform=(ax4.transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)), fontsize='medium', va='bottom')
        ax5.text(0.0, 1.0, 'e)', transform=(ax5.transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)), fontsize='medium', va='bottom')
        ax6.text(0.0, 1.0, 'f)', transform=(ax6.transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)), fontsize='medium', va='bottom')
        ax7.text(0.0, 1.0, 'g)', transform=(ax7.transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)), fontsize='medium', va='bottom')
        plt.suptitle(f"Speed distribution of system {params['system_name']}")
        plt.tight_layout()
        if save_plots:
            plt.savefig(f"./{params['res_path']}/speed_analysis/speed_wind_stages_{params['n_stages']}.png", bbox_inches='tight')
            plt.savefig(f"./{params['pdf_res_path']}/speed_analysis/speed_wind_stages_{params['n_stages']}.pdf", bbox_inches='tight')
        if show_plots:
            plt.show()
        else:
            plt.close()

            
        fig, (ax, ax1) = plt.subplots(2, 1, figsize = (8, 6), sharex = True, sharey = True)
        if len(params['blue_particle_idx']) > 0:
            ax.plot(params['window_center_sec'], mean_speed_b, 'b-',label = 'Blue droplets')
        if len(params['red_particle_idx']) > 0:
            ax.plot(params['window_center_sec'], mean_speed_r, 'r-', label = 'Red droplets')
        for i, frame in enumerate(params['frames_stages']):
            ax.bar(frame/params['fps'], height = 2000, width = params['window_length'], bottom = -10, color = params['stages_shades'][i], alpha = 0.5)
        ax.grid(linewidth = 0.2)
        ax.legend(fontsize = 10)
        ax.set_title(f"Velocity properties of system {params['system_name']}")
        if params['video_selection'] in ['25b25r_lowconc_1', '25b25r_lowconc_2', '25b25r_lowconc_3', '25b25r_lowconc_5', '25b25r_lowconc_6']:
            ax.set(ylim = (-0.1, 2.5), xlim = (-200, 14000))
        elif params['video_selection'] in ['25b25r-1', '25b25r-2']:
            ax.set(ylim = (-0.1, 4))
        elif params['video_selection'] in ['1b_&_1r_1', '1b_&_1r_2', '1b_&_1r_3']:
            ax.set(ylim = (-0.1, 40))
        else:
            ax.set(ylim = (-0.1, 2))
        ax.set(ylabel = r'$\langle v \rangle$ [mm/s]')
        ax.legend(loc = (0.1, 0.7), fontsize = 10)
        if len(params['blue_particle_idx']) > 0:
            ax1.plot(params['window_center_sec'], std_speed_b, 'b-')
        if len(params['red_particle_idx']) > 0:
            ax1.plot(params['window_center_sec'], std_speed_r, 'r-')
        for i, frame in enumerate(params['frames_stages']):
            ax1.bar(frame/params['fps'], height = 2000, width = params['window_length'], bottom = -10, color = params['stages_shades'][i], alpha = 0.5, label = f"Stage {i+1}")
        ax1.grid(linewidth = 0.2)
        ax1.legend(loc = (0.13, 0.4), fontsize = 10)
        ax1.set(xlabel = 'Window time [s]', ylabel = r'$std(v)$ [mm/s]')
        ax.text(0.0, 1.0, 'a)', transform=(ax.transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)), fontsize='medium', va='bottom')
        ax1.text(0.0, 1.0, 'b)', transform=(ax1.transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)), fontsize='medium', va='bottom')
        plt.tight_layout()
        if save_plots:
            plt.savefig(f"./{params['res_path']}/speed_analysis/speed_windowed_mean_std_{params['n_stages']}.png", bbox_inches='tight')
            plt.savefig(f"./{params['pdf_res_path']}/speed_analysis/speed_windowed_mean_std_{params['n_stages']}.pdf", bbox_inches='tight')
        if show_plots:
            plt.show()
        else:
            plt.close()


        fig, (ax, ax1) = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
        if len(params['blue_particle_idx']) > 0:
            ax.plot(params['window_center_sec'], r2_wind_b, 'b', label = 'Blue droplets')
            ax1.plot(params['window_center_sec'], fit_results_wind_b[:, 0], 'b', label = 'Blue droplets')
        if len(params['red_particle_idx']) > 0:
            ax.plot(params['window_center_sec'], r2_wind_r, 'r', label = 'Red droplets')
            ax1.plot(params['window_center_sec'], fit_results_wind_r[:, 0], 'r', label = 'Red droplets')
        ax.grid(linewidth = 0.2)
        ax.legend(fontsize = 10)
        ax.set(xlabel = 'Window time [s]', ylabel = r'$R^2$', title = r'$R^2$')
        
        if params['video_selection'] == '49b1r_post_merge':
            ax1.set(ylim = (0, .5))
        elif params['video_selection'] == '49b1r':
            ax1.set(ylim = (0, 4))
        elif params['video_selection'] == '25b25r-1':
            ax1.set(ylim = (0, 4))
        ax1.set(ylabel = r'$\sigma \; [mm/s]$', xlabel = 'Window time [s]', title = r'$\sigma$')
        ax1.legend(fontsize = 10)
        ax1.grid(linewidth = 0.2)
        ax.text(0.0, 1.0, 'a)', transform=(ax.transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)), fontsize='medium', va='bottom')
        ax1.text(0.0, 1.0, 'b)', transform=(ax1.transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)), fontsize='medium', va='bottom')
        plt.suptitle(f"Windowed Speed distribution with 2D Maxwell Boltzmann fit of system {params['system_name']}")
        plt.tight_layout()
        if save_plots: 
            plt.savefig(f"./{params['res_path']}/speed_analysis/speed_distribution_windowed.png", bbox_inches='tight')
            plt.savefig(f"./{params['pdf_res_path']}/speed_analysis/speed_distribution_windowed.pdf", bbox_inches='tight')
        if show_plots:
            plt.show()
        else:
            plt.close()

        fig, (ax, ax1, ax2) = plt.subplots(1, 3, figsize=(12, 4), sharex=True)
        if len(params['blue_particle_idx']):
            ax.plot(params['window_center_sec'], fit_results_wind_g_b[:, 0], 'b', label = 'Blue droplets')
            ax1.plot(params['window_center_sec'], fit_results_wind_g_b[:, 1], 'b', label = 'Blue droplets')
            ax2.plot(params['window_center_sec'], fit_results_wind_g_b[:, 2], 'b', label = 'Blue droplets')


        if len(params['red_particle_idx']):
            ax.plot(params['window_center_sec'], fit_results_wind_g_r[:, 0], 'r', label = 'Red droplets')
            ax1.plot(params['window_center_sec'], fit_results_wind_g_r[:, 1], 'r', label = 'Red droplets')
            ax2.plot(params['window_center_sec'], fit_results_wind_g_r[:, 2], 'r', label = 'Red droplets')

        ax.set(xlabel = 'Window time [s]', ylabel = r'$\sigma$')
        ax1.set(xlabel = 'Window time [s]', ylabel = r'$\beta$')
        ax2.set(xlabel = 'Window time [s]', ylabel = r'$A$')
        ax.grid(linewidth = 0.2)
        ax1.grid(linewidth = 0.2)
        ax2.grid(linewidth = 0.2)
        ax.legend(fontsize = 10)
        
        ax.text(0.0, 1.0, 'a)', transform=(ax.transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)), fontsize='medium', va='bottom')
        ax1.text(0.0, 1.0, 'b)', transform=(ax1.transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)), fontsize='medium', va='bottom')
        ax2.text(0.0, 1.0, 'c)', transform=(ax2.transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)), fontsize='medium', va='bottom')
        plt.suptitle(f"Generalized MB fit parameters evolution of system {params['system_name']}")
        plt.tight_layout()
        if save_plots: 
            plt.savefig(f"./{params['res_path']}/speed_analysis/fit_results_generalizedMB.png", bbox_inches='tight')
            plt.savefig(f"./{params['pdf_res_path']}/speed_analysis/fit_results_generalizedMB.pdf", bbox_inches='tight')
        if show_plots:
            plt.show()
        else:
            plt.close()


        if animated_plot_results:
            fig, (ax, ax1) = plt.subplots(1, 2, figsize = (10, 4), sharex = True, sharey = True)
            anim_running = True

            def update_plot(step):
                # update titles 
                title.set_text(f"Speed distribution of system {params['system_name']} at  " + r'$T_w$' + f" = {params['startFrames'][step]/params['fps'] + params['window_length']/2} s")
                
                if len(params['blue_particle_idx']) > 0: 
                    line_b.set_ydata(MB_2D(x_interval_for_fit, *fit_results_wind_b[step, :, 0]))
                    line_b1.set_ydata(MB_2D_generalized(x_interval_for_fit, *fit_results_wind_g_b[step, :, 0]))
                    for i, b in enumerate(bar_container_r):
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

            title = plt.suptitle(f"Speed distribution of system {params['system_name']} at  " + r'$T_w$' + f" = {params['startFrames'][0]/params['fps'] + params['window_length']/2} s")

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
            if params['video_selection'] in ['1b_&_1r_1', '1b_&_1r_2', '1b_&_1r_3']:
                ax1.set(xlabel = f"v [{params['speed_units']}]", ylim = (0, .35))
            else:
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
                

