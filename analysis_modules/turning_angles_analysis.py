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

def run_turning_analysis(trajectories, frames, EMSD_wind, pw_exp, maxLagtime_msd, params, show_plots, save_plots, run_analysis_verb, animated_plot_results):
    print('    Global turning angles analysis...')
    EMSD_wind_b, EMSD_wind_r = EMSD_wind[0], EMSD_wind[1]
    pw_exp_wind_b, pw_exp_wind_r = pw_exp[0], pw_exp[1]
    
    turn_angles_bins = np.linspace(-np.pi, np.pi, 601)
    turn_angles_bin_centers = turn_angles_bins[:-1] + np.diff(turn_angles_bins) / 2
    x_interval_for_fit_turn = np.linspace(turn_angles_bins[0], turn_angles_bins[-1], 10000)
    
    if len(params['blue_particle_idx']):
        blueTrajs = get_trajs(trajectories.loc[trajectories.particle.isin(params['blue_particle_idx'])], params['fps'], params['pxDimension'])
        theta_blue = ys.turning_angles_ensemble(blueTrajs, centered = True)
        turn_angles_b = np.histogram(theta_blue, bins = turn_angles_bins, density = True)[0]
        
        # fit turning angles distribution with normal distribution
        fit_results_gaussian_b, r2_blue  = fit_hist(turn_angles_b, turn_angles_bin_centers, normal_distr, [1., 0.], maxfev_ = 10000)
        print(f"        Gaussian fit           -- Blue droplets -- σ = {np.round(fit_results_gaussian_b[0, 0], 3)} ± {np.round(fit_results_gaussian_b[0, 1], 3)}, μ = {np.round(fit_results_gaussian_b[1, 0], 4)} ± {np.round(fit_results_gaussian_b[1, 1], 4)}, r2 = {np.round(r2_blue, 3)}")
        
        # fit turning angles distribution with lorentzian distribution
        fit_results_lorentzian_b, r2_blue = fit_hist(turn_angles_b, turn_angles_bin_centers, wrapped_lorentzian_distr, [1., 0.], maxfev_ = 10000)
        print(f"        Wrapped Lorentzian fit -- Blue droplets -- γ = {np.round(fit_results_lorentzian_b[0, 0], 3)} ± {np.round(fit_results_lorentzian_b[0, 1], 3)}, μ = {np.round(fit_results_lorentzian_b[1, 0], 4)} ± {np.round(fit_results_lorentzian_b[1, 1], 4)}, r2 = {np.round(r2_blue, 3)}")

    if len(params['red_particle_idx']):
        redTrajs = get_trajs(trajectories.loc[trajectories.particle.isin(params['red_particle_idx'])], params['fps'], params['pxDimension'])
        theta_red  = ys.turning_angles_ensemble(redTrajs, centered = True)
        turn_angles_r = np.histogram(theta_red, bins = turn_angles_bins, density = True)[0]
        
        # fit turning angles distribution with normal distribution
        fit_results_gaussian_r, r2_red  = fit_hist(turn_angles_r, turn_angles_bin_centers, normal_distr, [1., 0.], maxfev_ = 10000)
        print(f"        Gaussian fit           -- Red droplets  -- σ = {np.round(fit_results_gaussian_r[0, 0], 3)} ± {np.round(fit_results_gaussian_r[0, 1], 3)}, μ = {np.round(fit_results_gaussian_r[1, 0], 4)} ± {np.round(fit_results_gaussian_r[1, 1], 4)}, r2 = {np.round(r2_red, 3)}")
        
        # fit turning angles distribution with lorentzian distribution
        fit_results_lorentzian_r, r2_red = fit_hist(turn_angles_r, turn_angles_bin_centers, wrapped_lorentzian_distr, [1., 0.], maxfev_ = 10000)
        print(f"        Wrapped Lorentzian fit -- Red droplets  -- γ = {np.round(fit_results_lorentzian_r[0, 0], 3)} ± {np.round(fit_results_lorentzian_r[0, 1], 3)}, μ = {np.round(fit_results_lorentzian_r[1, 0], 4)} ± {np.round(fit_results_lorentzian_r[1, 1], 4)}, r2 = {np.round(r2_red, 3)}")
        
    if 1:
        fig, (ax, ax1) = plt.subplots(1, 2, figsize=(12, 4), sharex=True, sharey=True)
        if len(params['blue_particle_idx']) > 0:
            ax.bar(turn_angles_bin_centers, turn_angles_b, width = np.diff(turn_angles_bins)[0], color = 'b', alpha = 0.5, label = 'Blue droplets')
            ax.plot(x_interval_for_fit_turn, normal_distr(x_interval_for_fit_turn, *fit_results_gaussian_b[:, 0]), label = 'Gaussian fit')
            ax.plot(x_interval_for_fit_turn, wrapped_lorentzian_distr(x_interval_for_fit_turn, *fit_results_lorentzian_b[:, 0]), label = 'Lorentzian fit')
        ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], [r'-$\pi$', r'$-\frac{\pi}{2}$', '$0$', r'$\frac{\pi}{2}$', r'$\pi$'])
        ax.set(ylabel='pdf', xlabel= r'$\theta$ [rad]')
        ax.legend(fontsize = 10)
        if params['video_selection'] not in ['1b_&_1r_1', '1b_&_1r_2', '1b_&_1r_3']:
            ax.set_ylim(0, 6)
        ax.grid(linewidth = 0.2)
        if len(params['red_particle_idx']) > 0:
            ax1.bar(turn_angles_bin_centers, turn_angles_r, width = np.diff(turn_angles_bins)[0], color = 'r', alpha = 0.5, label = 'Red droplets')
            ax1.plot(x_interval_for_fit_turn, normal_distr(x_interval_for_fit_turn, *fit_results_gaussian_r[:, 0]), label = 'Gaussian fit')
            ax1.plot(x_interval_for_fit_turn, wrapped_lorentzian_distr(x_interval_for_fit_turn, *fit_results_lorentzian_r[:, 0]), label = 'Lorentzian fit')
        ax1.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi],[r'-$\pi$', r'$-\frac{\pi}{2}$', '$0$', r'$\frac{\pi}{2}$', r'$\pi$'])
        ax1.set(xlim = (-np.pi/4, np.pi/4))
        ax1.set(xlabel= r'$\theta$ [rad]')
        ax1.legend(fontsize = 10)
        ax1.grid(linewidth = 0.2)
        ax.text(0.0, 1.0, 'a)', transform=(ax.transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)), fontsize='medium', va='bottom')
        ax1.text(0.0, 1.0, 'b)', transform=(ax1.transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)), fontsize='medium', va='bottom')
        plt.suptitle(f"Turning angles pdf - Lorentzian fit of system {params['system_name']}")
        plt.tight_layout()
        if save_plots: 
            plt.savefig(f"./{params['res_path']}/turning_angles_analysis/turn_ang.png", bbox_inches='tight')
            plt.savefig(f"./{params['pdf_res_path']}/turning_angles_analysis/turn_ang.pdf", bbox_inches='tight')
        if show_plots:
            plt.show()
        else:
            plt.close()
    
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
            
    t_r_blue = 1/( (1 - np.exp(-lorentzian_fit_results_wind_b[:, 0, 0]))/(2*1/params['fps']) )
    t_r_red = 1/( (1 - np.exp(-lorentzian_fit_results_wind_r[:, 0, 0]))/(2*1/params['fps']) )

    if 1: 		
        fig, axs = plt.subplots(1, params['n_stages'], figsize = (15, 4), sharex = True, sharey = True)
        for i, step in enumerate(params['steps_plot']):
            if len(params['blue_particle_idx']) > 0:
                axs[i].bar(turn_angles_bin_centers, turn_angles_b[step], width = np.diff(turn_angles_bins)[0], color = 'b', alpha = 0.5, label = 'Blue droplets')
                #axs[i].plot(x, kde_blue_turn[i], color='blue')
            
            if len(params['red_particle_idx']) > 0:
                axs[i].bar(turn_angles_bin_centers, turn_angles_r[step], width = np.diff(turn_angles_bins)[0], color = 'r', alpha = 0.5, label = 'Red droplets')
                #axs[i].plot(x, kde_red_turn[i], color='red')

            axs[i].grid(linewidth = 0.2)
            axs[i].set(title = f"Stage {i + 1}", xlabel = r'$\theta$ [rad]')
            axs[i].text(0.0, 1.0, f"{params['letter_labels'][i]}", transform=(axs[i].transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)), fontsize='medium', va='bottom')
        axs[0].set_xticks([-np.pi, -np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2, np.pi], [r'-$\pi$', r'$-\pi/2$', r'$-\pi/4$', '$0$', r'$\pi/4$', r'$\pi/2$', r'$\pi$'])
        axs[0].set(ylabel = 'pdf [1/rad]', xlim = (-np.pi/4, np.pi/4))
        if params['video_selection'] == '1b_&_1r_1':
            axs[0].set(ylim = (0, 15))
        else:
            axs[0].set(ylim = (0, 12))
        axs[-1].legend(fontsize = 10)
        plt.suptitle(f"Turning angles distribution of system {params['system_name']}")
        plt.tight_layout()
        if save_plots:
            plt.savefig(f"./{params['res_path']}/turning_angles_analysis/turning_angles_wind_stages_{params['n_stages']}_kde.png", bbox_inches='tight')
            plt.savefig(f"./{params['pdf_res_path']}/turning_angles_analysis/turning_angles_wind_stages_{params['n_stages']}_kde.pdf", bbox_inches='tight')
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
        ax1.set(title = f"Stage {i + 1}", xlabel = r'$\theta$ [rad]', ylabel = 'pdf [1/rad]', xlim = (-np.pi/4, np.pi/4))
        if params['video_selection'] == '1b_&_1r_1':
            ax1.set(ylim = (0, 15))
        else:
            ax1.set(ylim = (0, 14))
        i, step = 1, params['steps_plot'][1]
        ax2 = fig.add_subplot(gs[0, 2:4], sharex = ax1, sharey = ax1)
        if len(params['blue_particle_idx']) > 0:
            ax2.bar(turn_angles_bin_centers, turn_angles_b[step], width = np.diff(turn_angles_bins)[0], color = 'b', alpha = 0.5, label = 'Blue droplets')
            ax2.plot(x_interval_for_fit_turn, wrapped_lorentzian_distr(x_interval_for_fit_turn, *lorentzian_fit_results_wind_b[step, :, 0]), color = 'blue', label = 'Wrapped Lorentzian fit')
        if len(params['red_particle_idx']) > 0:
            ax2.bar(turn_angles_bin_centers, turn_angles_r[step], width = np.diff(turn_angles_bins)[0], color = 'r', alpha = 0.5, label = 'Red droplets')
            ax2.plot(x_interval_for_fit_turn, wrapped_lorentzian_distr(x_interval_for_fit_turn, *lorentzian_fit_results_wind_r[step, :, 0]), color = 'red', label = 'Wrapped Lorentzian fit')
        ax2.grid(linewidth = 0.2)
        ax2.set(title = f"Stage {i + 1}", xlabel = r'$\theta$ [rad]')
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
        ax3.set(title = f"Stage {i + 1}", xlabel = r'$\theta$ [rad]')
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
        ax4.set(title = f"Stage {i + 1}", xlabel = r'$\theta$ [rad]')
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
        ax5.set(title = f"Stage {i + 1}", xlabel = r'$\theta$ [rad]')
        ax5.legend(fontsize = 10)
        plt.setp(ax5.get_yticklabels(), visible=False)
        ax6 = fig.add_subplot(gs[1, :5])
        if len(params['blue_particle_idx']) > 0:
            ax6.plot(params['window_center_sec'], lorentzian_fit_results_wind_b[:, 0, 0], 'b')
        if len(params['red_particle_idx']) > 0:
            ax6.plot(params['window_center_sec'], lorentzian_fit_results_wind_r[:, 0, 0], 'r')
            
        for i, frame in enumerate(params['frames_stages']):
            ax6.bar(frame/params['fps'], 20000, params['window_length'], bottom = -100, color = params['stages_shades'][i], alpha = 0.5, label = f"Stage {i+1}")
        ax6.set(ylabel = r'$\gamma \; [rad]$', xlabel = 'Window time [s]', title = 'Scale factor')
        if params['video_selection'] in ['25b25r_lowconc_1', '25b25r_lowconc_2', '25b25r_lowconc_3', '25b25r_lowconc_5', '25b25r_lowconc_6']:
            ax6.set(ylim = (0, 0.2), xlim = (-200, 14000))
        elif params['video_selection'] in ['1b_&_1r_1', '1b_&_1r_2', '1b_&_1r_3']:
            ax6.set(ylim = (0, 0.12))
        else:
            ax6.set(ylim = (0, 0.3))
        ax6.grid(linewidth = 0.2)
        ax6.legend(['Blue droplets', 'Red droplets'], fontsize = 10, loc = (0.09, 0.7))
        ax7 = fig.add_subplot(gs[1, 5:])
        if len(params['blue_particle_idx']) > 0:
            ax7.plot(params['window_center_sec'], lorentzian_fit_results_wind_b[:, 1, 0], 'b')
        if len(params['red_particle_idx']) > 0:
            ax7.plot(params['window_center_sec'], lorentzian_fit_results_wind_r[:, 1, 0], 'r')
            
        for i, frame in enumerate(params['frames_stages']):
            ax7.bar(frame/params['fps'], 20000, params['window_length'], bottom = -100, color = params['stages_shades'][i], alpha = 0.5, label = f"Stage {i+1}")
        ax7.set(ylabel = r'$\mu \; [rad]$', xlabel = 'Window time [s]', title = 'Mean')
        if params['video_selection'] in ['25b25r_lowconc_1', '25b25r_lowconc_2', '25b25r_lowconc_3', '25b25r_lowconc_5', '25b25r_lowconc_6']:
            ax7.set(ylim = (-0.01, 0.01), xlim = (-200, 14000))
        elif params['video_selection'] in ['1b_&_1r_1', '1b_&_1r_2', '1b_&_1r_3']:  
            ax7.set(ylim = (-0.5, 0.5))
        else:
            ax7.set(ylim = (-0.01, 0.01))
        #ax7.legend(loc = (0.1, 0.4), fontsize = 10)
        ax7.grid(linewidth = 0.2)
        ax1.text(0.0, 1.0, 'a)', transform=(ax1.transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)), fontsize='medium', va='bottom')
        ax2.text(0.0, 1.0, 'b)', transform=(ax2.transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)), fontsize='medium', va='bottom')
        ax3.text(0.0, 1.0, 'c)', transform=(ax3.transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)), fontsize='medium', va='bottom')
        ax4.text(0.0, 1.0, 'd)', transform=(ax4.transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)), fontsize='medium', va='bottom')
        ax5.text(0.0, 1.0, 'e)', transform=(ax5.transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)), fontsize='medium', va='bottom')
        ax6.text(0.0, 1.0, 'f)', transform=(ax6.transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)), fontsize='medium', va='bottom')
        ax7.text(0.0, 1.0, 'g)', transform=(ax7.transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)), fontsize='medium', va='bottom')
        plt.suptitle(f"Turning angles distribution of system {params['system_name']}")
        plt.tight_layout()
        if save_plots:
            plt.savefig(f"./{params['res_path']}/turning_angles_analysis/turning_angles_wind_stages_{params['n_stages']}.png", bbox_inches='tight')
            plt.savefig(f"./{params['pdf_res_path']}/turning_angles_analysis/turning_angles_wind_stages_{params['n_stages']}.pdf", bbox_inches='tight')
        if show_plots:
            plt.show()
        else:
            plt.close()
        
        fig, ax = plt.subplots(1, 1, figsize = (10, 4))
        for i, frame in enumerate(params['frames_stages']):
            ax.bar(frame/params['fps'], 20000, params['window_length'], bottom = -100, color = params['stages_shades'][i], alpha = 0.5)
        ax.set(ylim = (0, 8), xlabel = 'Window time [s]', ylabel = r'$\tau_r \; [s]$', title = f"Relaxation time of system {params['system_name']}")
        if len(params['blue_particle_idx']) > 0:
            ax.plot(params['window_center_sec'], t_r_blue, 'b-', label = 'Blue droplets')
        if len(params['red_particle_idx']) > 0:
            ax.plot(params['window_center_sec'], t_r_red, 'r-', label = 'Red droplets')
        ax.legend(fontsize = 10)
        ax.grid(linewidth = 0.2)
        if params['video_selection'] in ['25b25r_lowconc_1', '25b25r_lowconc_2', '25b25r_lowconc_3', '25b25r_lowconc_5', '25b25r_lowconc_6']:
            ax.set(ylim = (0, 4), xlim = (-200, 14000))
        elif params['video_selection'] in ['1b_&_1r_1', '1b_&_1r_2', '1b_&_1r_3']:
            ax.set(ylim = (0, 15))
        else:
            ax.set(ylim = (0, 4))
        if save_plots:
            plt.savefig(f"./{params['res_path']}/turning_angles_analysis/relaxation_time.png", bbox_inches='tight')
            plt.savefig(f"./{params['pdf_res_path']}/turning_angles_analysis/relaxation_time.pdf", bbox_inches='tight')
        if show_plots:
            plt.show()
        else:
            plt.close()        

        fig, (ax, ax1) = plt.subplots(2, 1, figsize = (8, 6), sharex = True)
        if len(params['blue_particle_idx']) > 0:
            ax.plot(params['window_center_sec'], lorentzian_fit_results_wind_b[:, 1, 0], 'b', label = 'Blue droplets')
        if len(params['red_particle_idx']) > 0:
            ax.plot(params['window_center_sec'], lorentzian_fit_results_wind_r[:, 1, 0], 'r', label = 'Red droplets')
        ax.plot(params['window_center_sec'], np.zeros(params['n_windows']), 'k-')
        ax.set(ylabel = r'$x_0 \; [rad]$', title = f"Wrapped lorentzian fit of system {params['system_name']}")
        ax.legend(loc = (0.09, 0.7), fontsize = 10)
        ax.grid(linewidth = 0.2)
        for i, frame in enumerate(params['frames_stages']):
            ax.bar(frame/params['fps'], 20000, params['window_length'], bottom = -100, color = params['stages_shades'][i], alpha = 0.5)
        if params['video_selection'] in ['1b_&_1r_1', '1b_&_1r_2', '1b_&_1r_3']:
            ax.set_ylim(-0.3, 0.3)
        else:
            ax.set_ylim(-0.007, 0.007)
        
        ax.ticklabel_format(axis = 'y', style = 'sci', scilimits = (0, 0))
        if len(params['blue_particle_idx']) > 0:
            ax1.plot(params['window_center_sec'], lorentzian_fit_results_wind_b[:, 0, 0], 'b')
        if len(params['red_particle_idx']) > 0:
            ax1.plot(params['window_center_sec'], lorentzian_fit_results_wind_r[:, 0, 0], 'r')
        for i, frame in enumerate(params['frames_stages']):
            ax1.bar(frame/params['fps'], 20000, params['window_length'], bottom = -100, color = params['stages_shades'][i], alpha = 0.5, label = f"Stage {i+1}")
        ax1.set(ylabel = r'$\gamma \; [rad]$', xlabel = 'Window time [s]')
        if params['video_selection'] in ['25b25r_lowconc_1', '25b25r_lowconc_2', '25b25r_lowconc_3', '25b25r_lowconc_5', '25b25r_lowconc_6']:
            ax1.set(ylim = (0, 0.3), xlim = (-200, 14000))
        elif params['video_selection'] in ['1b_&_1r_1', '1b_&_1r_2', '1b_&_1r_3']:
            ax1.set(ylim = (0, 0.15))
        else:
            ax1.set(ylim = (0, 0.3))
        ax1.grid(linewidth = 0.2)
        ax1.legend(loc = (0.1, 0.4), fontsize = 10)
        ax.text(0.0, 1.0, 'a)', transform=(ax.transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)), fontsize='medium', va='bottom')
        ax1.text(0.0, 1.0, 'b)', transform=(ax1.transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)), fontsize='medium', va='bottom')
        plt.tight_layout()
        if save_plots: 
            plt.savefig(f"./{params['res_path']}/turning_angles_analysis/turn_ang_lorentzian_wind_stages_{params['n_stages']}.png", bbox_inches='tight')
            plt.savefig(f"./{params['pdf_res_path']}/turning_angles_analysis/turn_ang_lorentzian_wind_stages_{params['n_stages']}.pdf", bbox_inches='tight')
        if show_plots:
            plt.show()
        else:
            plt.close()

        fig, ax = plt.subplots(1, 1, figsize = (10, 4))
        if len(params['blue_particle_idx']) > 0:
            ax.plot(params['window_center_sec'], lorentzian_r2_wind_b, 'b', label = 'Lorentzian Fit')
            ax.plot(params['window_center_sec'], gaussian_r2_wind_b, 'b--', label = 'Gaussian Fit')

        if len(params['red_particle_idx']) > 0:
            ax.plot(params['window_center_sec'], gaussian_r2_wind_r, 'r--', label = 'Gaussian Fit')
            ax.plot(params['window_center_sec'], lorentzian_r2_wind_r, 'r', label = 'Lorentzian fit') 
        
        ax.grid(linewidth = 0.2)
        ax.legend(fontsize = 10)
        
        ax.set(xlabel = 'Window time [s]', ylabel = r'$R^2$', title = f"R² confront fit of the turning angles distribution of system {params['system_name']}")
        if save_plots: 
            plt.savefig(f"./{params['res_path']}/turning_angles_analysis/r2_confront.png", bbox_inches='tight')
            plt.savefig(f"./{params['pdf_res_path']}/turning_angles_analysis/r2_confront.pdf", bbox_inches='tight')
        if show_plots:
            plt.show()
        else:
            plt.close()
            
        if EMSD_wind_b is not None or EMSD_wind_r is not None:
            fig, axs = plt.subplots(1, 5, figsize = (18, 4), sharex = True, sharey = True)
            for i, step in enumerate(params['steps_plot']):
                if len(params['blue_particle_idx']) > 0:
                    axs[i].plot(np.arange(1, maxLagtime_msd + 1, 1)/params['fps'], EMSD_wind_b[0, step], color = params['shades_of_blue'][2], label = 'Blue droplets')
                    axs[i].fill_between(np.arange(1, maxLagtime_msd + 1, 1)/params['fps'], EMSD_wind_b[0, step] - 2/np.sqrt(len(params['blue_particle_idx'])) * EMSD_wind_b[1, step],\
                                        EMSD_wind_b[0, step] + 2/np.sqrt(len(params['blue_particle_idx'])) * EMSD_wind_b[1, step], alpha=0.5, edgecolor='#00FFFF', facecolor='#F0FFFF')
                    axs[i].axvline(t_r_blue[step], color = 'blue', linestyle = '--', label = r'$\tau_r$')
                if len(params['red_particle_idx']) > 0:
                    axs[i].plot(np.arange(1, maxLagtime_msd + 1, 1)/params['fps'], EMSD_wind_r[0, step], color = params['shades_of_red'][2], label = 'Red droplets')
                    axs[i].fill_between(np.arange(1, maxLagtime_msd + 1, 1)/params['fps'], EMSD_wind_r[0, step] - 2/np.sqrt(len(params['red_particle_idx'])) * EMSD_wind_r[1, step],\
                                        EMSD_wind_r[0, step] + 2/np.sqrt(len(params['red_particle_idx'])) * EMSD_wind_r[1, step], alpha=0.5, edgecolor='#FF0000', facecolor='#FF5A52')
                    axs[i].axvline(t_r_red[step], color = 'red', linestyle = '--', label = r'$\tau_r$')
                
                axs[i].grid(linewidth = 0.2)
                axs[i].set(xscale = 'log', yscale = 'log', title = f"Stage {i + 1}", xlabel = 'Lag time [s]')
                axs[i].text(0.0, 1.0, f"{params['letter_labels'][i]}", transform=(axs[i].transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)), fontsize='medium', va='bottom')
            axs[0].set(ylabel = r'$\langle \Delta r^2 \rangle$ [$mm^2$]')
            axs[0].legend(['Blue droplets', 'Red droplets'], fontsize = 10)
            axs[-1].legend([r'$\tau_r$', r'$\tau_r$'], fontsize = 10)
            plt.suptitle(f"EMSD of system {params['system_name']}")
            plt.tight_layout()
            if save_plots:
                plt.savefig(f"./{params['res_path']}/tamsd_analysis/EMSD_relaxation_time.png", bbox_inches='tight')
                plt.savefig(f"./{params['pdf_res_path']}/tamsd_analysis/EMSD_relaxation_time.pdf", bbox_inches='tight')
            if show_plots:
                plt.show()
            else:
                plt.close()
                
            
            gs = gridspec.GridSpec(2, 10)
            fig = plt.figure(figsize = (18, 6))
            i, step = 0, params['steps_plot'][0]
            ax1 = fig.add_subplot(gs[0, :2])
            if len(params['blue_particle_idx']) > 0:
                ax1.plot(np.arange(1, maxLagtime_msd + 1, 1)/params['fps'], EMSD_wind_b[0, step], color = 'b', label = 'Blue droplets')
                ax1.fill_between(np.arange(1, maxLagtime_msd + 1, 1)/params['fps'], EMSD_wind_b[0, step] - 2/np.sqrt(len(params['blue_particle_idx'])) * EMSD_wind_b[1, step],\
                                EMSD_wind_b[0, step] + 2/np.sqrt(len(params['blue_particle_idx'])) * EMSD_wind_b[1, step], alpha=0.5, edgecolor='#00FFFF', facecolor='#F0FFFF')
                ax1.axvline(t_r_blue[step], color = 'blue', linestyle = '--', label = r'$\tau_b$')
            if len(params['red_particle_idx']) > 0:
                ax1.plot(np.arange(1, maxLagtime_msd + 1, 1)/params['fps'], EMSD_wind_r[0, step], color = 'r', label = 'Red droplets')
                ax1.fill_between(np.arange(1, maxLagtime_msd + 1, 1)/params['fps'], EMSD_wind_r[0, step] - 2/np.sqrt(len(params['red_particle_idx'])) * EMSD_wind_r[1, step],\
                                EMSD_wind_r[0, step] + 2/np.sqrt(len(params['red_particle_idx'])) * EMSD_wind_r[1, step], alpha=0.5, edgecolor='#FF0000', facecolor='#FF5A52')
                ax1.axvline(t_r_red[step], color = 'red', linestyle = '--', label = r'$\tau_r$')
            
            ax1.grid(linewidth = 0.2)
            ax1.set(xscale = 'log', yscale = 'log', title = f"Stage {i + 1}", xlabel = 'Lag time [s]', ylabel = r'$\langle \Delta r^2 \rangle$ [$mm^2$]')
            #ax1.legend(['Blue droplets', 'Red droplets'], fontsize = 10)
            i, step = 1, params['steps_plot'][1]
            ax2 = fig.add_subplot(gs[0, 2:4], sharex = ax1, sharey = ax1)
            if len(params['blue_particle_idx']) > 0:
                ax2.plot(np.arange(1, maxLagtime_msd + 1, 1)/params['fps'], EMSD_wind_b[0, step], color = 'b', label = 'Blue droplets')
                ax2.fill_between(np.arange(1, maxLagtime_msd + 1, 1)/params['fps'], EMSD_wind_b[0, step] - 2/np.sqrt(len(params['blue_particle_idx'])) * EMSD_wind_b[1, step],\
                                EMSD_wind_b[0, step] + 2/np.sqrt(len(params['blue_particle_idx'])) * EMSD_wind_b[1, step], alpha=0.5, edgecolor='#00FFFF', facecolor='#F0FFFF')
                ax2.axvline(t_r_blue[step], color = 'blue', linestyle = '--', label = r'$\tau_b$')
            if len(params['red_particle_idx']) > 0:
                ax2.plot(np.arange(1, maxLagtime_msd + 1, 1)/params['fps'], EMSD_wind_r[0, step], color = 'r', label = 'Red droplets')
                ax2.fill_between(np.arange(1, maxLagtime_msd + 1, 1)/params['fps'], EMSD_wind_r[0, step] - 2/np.sqrt(len(params['red_particle_idx'])) * EMSD_wind_r[1, step],\
                                EMSD_wind_r[0, step] + 2/np.sqrt(len(params['red_particle_idx'])) * EMSD_wind_r[1, step], alpha=0.5, edgecolor='#FF0000', facecolor='#FF5A52')
                ax2.axvline(t_r_red[step], color = 'red', linestyle = '--', label = r'$\tau_r$')
            ax2.grid(linewidth = 0.2)
            ax2.set(xscale = 'log', yscale = 'log', title = f"Stage {i + 1}", xlabel = 'Lag time [s]')
            plt.setp(ax2.get_yticklabels(), visible=False)
            i, step = 2, params['steps_plot'][2]
            ax3 = fig.add_subplot(gs[0, 4:6], sharex = ax1, sharey = ax1)
            if len(params['blue_particle_idx']) > 0:
                ax3.plot(np.arange(1, maxLagtime_msd + 1, 1)/params['fps'], EMSD_wind_b[0, step], color = 'b', label = 'Blue droplets')
                ax3.fill_between(np.arange(1, maxLagtime_msd + 1, 1)/params['fps'], EMSD_wind_b[0, step] - 2/np.sqrt(len(params['blue_particle_idx'])) * EMSD_wind_b[1, step],\
                                EMSD_wind_b[0, step] + 2/np.sqrt(len(params['blue_particle_idx'])) * EMSD_wind_b[1, step], alpha=0.5, edgecolor='#00FFFF', facecolor='#F0FFFF')
                ax3.axvline(t_r_blue[step], color = 'blue', linestyle = '--', label = r'$\tau_b$')
            if len(params['red_particle_idx']) > 0:
                ax3.plot(np.arange(1, maxLagtime_msd + 1, 1)/params['fps'], EMSD_wind_r[0, step], color = 'r', label = 'Red droplets')
            
                ax3.fill_between(np.arange(1, maxLagtime_msd + 1, 1)/params['fps'], EMSD_wind_r[0, step] - 2/np.sqrt(len(params['red_particle_idx'])) * EMSD_wind_r[1, step],\
                                EMSD_wind_r[0, step] + 2/np.sqrt(len(params['red_particle_idx'])) * EMSD_wind_r[1, step], alpha=0.5, edgecolor='#FF0000', facecolor='#FF5A52')
                ax3.axvline(t_r_red[step], color = 'red', linestyle = '--', label = r'$\tau_r$')
            ax3.grid(linewidth = 0.2)
            ax3.set(xscale = 'log', yscale = 'log', title = f"Stage {i + 1}", xlabel = 'Lag time [s]')
            plt.setp(ax3.get_yticklabels(), visible=False)
            i, step = 3, params['steps_plot'][3]
            ax4 = fig.add_subplot(gs[0, 6:8], sharex = ax1, sharey = ax1)
            if len(params['blue_particle_idx']) > 0:
                ax4.plot(np.arange(1, maxLagtime_msd + 1, 1)/params['fps'], EMSD_wind_b[0, step], color = 'b', label = 'Blue droplets')
                ax4.fill_between(np.arange(1, maxLagtime_msd + 1, 1)/params['fps'], EMSD_wind_b[0, step] - 2/np.sqrt(len(params['blue_particle_idx'])) * EMSD_wind_b[1, step],\
                                EMSD_wind_b[0, step] + 2/np.sqrt(len(params['blue_particle_idx'])) * EMSD_wind_b[1, step], alpha=0.5, edgecolor='#00FFFF', facecolor='#F0FFFF')
                ax4.axvline(t_r_blue[step], color = 'blue', linestyle = '--', label = r'$\tau_b$')
    
            if len(params['red_particle_idx']) > 0:
                ax4.plot(np.arange(1, maxLagtime_msd + 1, 1)/params['fps'], EMSD_wind_r[0, step], color = 'r', label = 'Red droplets')
                ax4.fill_between(np.arange(1, maxLagtime_msd + 1, 1)/params['fps'], EMSD_wind_r[0, step] - 2/np.sqrt(len(params['red_particle_idx'])) * EMSD_wind_r[1, step],\
                                EMSD_wind_r[0, step] + 2/np.sqrt(len(params['red_particle_idx'])) * EMSD_wind_r[1, step], alpha=0.5, edgecolor='#FF0000', facecolor='#FF5A52')
                ax4.axvline(t_r_red[step], color = 'red', linestyle = '--', label = r'$\tau_r$')
            
            ax4.grid(linewidth = 0.2)
            ax4.set(xscale = 'log', yscale = 'log', title = f"Stage {i + 1}", xlabel = 'Lag time [s]')
            plt.setp(ax4.get_yticklabels(), visible=False)
            i, step = 4, params['steps_plot'][4]
            ax5 = fig.add_subplot(gs[0, 8:10], sharex = ax1, sharey = ax1)
            if len(params['blue_particle_idx']) > 0:
                ax5.axvline(t_r_blue[step], color = 'b', linestyle = '--', label = r'$\tau_b$')
                ax5.plot(np.arange(1, maxLagtime_msd + 1, 1)/params['fps'], EMSD_wind_b[0, step], color = 'b', label = 'Blue droplets')
                ax5.fill_between(np.arange(1, maxLagtime_msd + 1, 1)/params['fps'], EMSD_wind_b[0, step] - 2/np.sqrt(len(params['blue_particle_idx'])) * EMSD_wind_b[1, step],\
                                EMSD_wind_b[0, step] + 2/np.sqrt(len(params['blue_particle_idx'])) * EMSD_wind_b[1, step], alpha=0.5, edgecolor='#00FFFF', facecolor='#F0FFFF')
            if len(params['red_particle_idx']) > 0:
                ax5.axvline(t_r_red[step], color = 'r', linestyle = '--', label = r'$\tau_r$')
                ax5.plot(np.arange(1, maxLagtime_msd + 1, 1)/params['fps'], EMSD_wind_r[0, step], color = 'r', label = 'Red droplets')
                ax5.fill_between(np.arange(1, maxLagtime_msd + 1, 1)/params['fps'], EMSD_wind_r[0, step] - 2/np.sqrt(len(params['red_particle_idx'])) * EMSD_wind_r[1, step],\
                                EMSD_wind_r[0, step] + 2/np.sqrt(len(params['red_particle_idx'])) * EMSD_wind_r[1, step], alpha=0.5, edgecolor='#FF0000', facecolor='#FF5A52')
            ax5.grid(linewidth = 0.2)
            ax5.set(xscale = 'log', yscale = 'log', title = f"Stage {i + 1}", xlabel = 'Lag time [s]')
            ax5.legend([r'$\tau_b$', r'$\tau_r$'], fontsize = 10)
            plt.setp(ax5.get_yticklabels(), visible=False)
            ax6 = fig.add_subplot(gs[1, :5])
            if len(params['blue_particle_idx']) > 0:
                ax6.plot(params['window_center_sec'], pw_exp_wind_b[:, 0, 1], 'b-', label = 'Blue droplets')
            if len(params['red_particle_idx']) > 0:
                ax6.plot(params['window_center_sec'], pw_exp_wind_r[:, 0, 1], 'r-', label = 'Red droplets ')
            ax6.plot(params['window_center_sec'], np.ones(params['n_windows']), 'k-')
            ax6.set(xlabel = 'Window time [s]', ylabel = r'$\alpha$', ylim = (-0.1, 2.1), title = 'Scaling exponents')
            ax6.legend(loc = (0.09, 0.7), fontsize = 10)
            ax6.grid(linewidth = 0.2)
            for i, frame in enumerate(params['frames_stages']):
                ax6.bar(frame/params['fps'], 2000, params['window_length'], bottom = -100, color = params['stages_shades'][i], alpha = 0.5)
            ax7 = fig.add_subplot(gs[1, 5:], sharex = ax6)
            if len(params['blue_particle_idx']) > 0:
                ax7.plot(params['window_center_sec'], pw_exp_wind_b[:, 0, 0], 'b-')
            if len(params['red_particle_idx']) > 0:
                ax7.plot(params['window_center_sec'], pw_exp_wind_r[:, 0, 0], 'r-')
            for i, frame in enumerate(params['frames_stages']):
                ax7.bar(frame/params['fps'], 2000, params['window_length'], bottom = -100, color = params['stages_shades'][i], alpha = 0.5, label = f"Stage {i+1}")
            if params['video_selection'] in ['25b25r_lowconc_1', '25b25r_lowconc_2', '25b25r_lowconc_3', '25b25r_lowconc_5', '25b25r_lowconc_6']:
                ax7.set(ylim=(-1, 20), xlim = (-200, 14000))
            elif params['video_selection'] in ['25b25r-1', '25b25r-2']:
                ax7.set(ylim=(-1, 200))
            elif params['video_selection'] in ['1b_&_1r_1', '1b_&_1r_2', '1b_&_1r_3']:
                ax7.set(ylim=(0, 1800))
            else:
                ax7.set(ylim=(-1, 10))
            #ax7.legend(loc = (0.6, 0.35), fontsize = 10)
            ax7.set(xlabel = 'Window time [s]', ylabel =r'$K{_\alpha} \; [mm^2/s^\alpha]$', title = 'Generalized diffusion coefficients')
            ax7.grid(linewidth = 0.2)
            ax1.text(0.0, 1.0, 'a)', transform=(ax1.transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)), fontsize='medium', va='bottom')
            ax2.text(0.0, 1.0, 'b)', transform=(ax2.transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)), fontsize='medium', va='bottom')
            ax3.text(0.0, 1.0, 'c)', transform=(ax3.transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)), fontsize='medium', va='bottom')
            ax4.text(0.0, 1.0, 'd)', transform=(ax4.transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)), fontsize='medium', va='bottom')
            ax5.text(0.0, 1.0, 'e)', transform=(ax5.transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)), fontsize='medium', va='bottom')
            ax6.text(0.0, 1.0, 'f)', transform=(ax6.transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)), fontsize='medium', va='bottom')
            ax7.text(0.0, 1.0, 'g)', transform=(ax7.transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)), fontsize='medium', va='bottom')
            plt.suptitle(f"EMSD of system {params['system_name']}")
            plt.tight_layout()
            if save_plots:
                plt.savefig(f"./{params['res_path']}/tamsd_analysis/EMSD_relaxation_time_v2.png", bbox_inches='tight')
                plt.savefig(f"./{params['pdf_res_path']}/tamsd_analysis/EMSD_relaxation_time_v2.pdf", bbox_inches='tight')
            if show_plots:
                plt.show()
            else:
                plt.close()

        if animated_plot_results:
            fig, (ax, ax1) = plt.subplots(2, 1, figsize=(8, 5), sharex = True, sharey = True)
            anim_running = True
            def update_plot(frame):
                title.set_text(f"Turning angles distribution of system {params['system_name']} at  " + r'$T_w$' + f"= {params['startFrames'][frame]/params['fps'] + params['window_length']/2} s")
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
            
            title = ax.set_title(f"Turning angles distribution of system {params['system_name']} at  " + r'$T_w$' + f"= {params['startFrames'][0]/params['fps'] + params['window_length']/2} s")

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
        return (turn_angles_b, turn_angles_r), (lorentzian_fit_results_wind_b, lorentzian_fit_results_wind_r), (turn_angles_bins, turn_angles_bin_centers, x_interval_for_fit_turn)
    elif (len(params['blue_particle_idx']) > 0) & (len(params['red_particle_idx']) == 0):
        return (turn_angles_b, None), (lorentzian_fit_results_wind_b, None), (turn_angles_bins, turn_angles_bin_centers, x_interval_for_fit_turn)
    elif (len(params['blue_particle_idx']) == 0) & (len(params['red_particle_idx']) > 0):
        return (None, turn_angles_r), (None, lorentzian_fit_results_wind_r), (turn_angles_bins, turn_angles_bin_centers, x_interval_for_fit_turn)
    