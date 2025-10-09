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

from analysis_utils import get_imsd, get_emsd, get_emsd_windowed_v2, powerLaw, onClick

def run_tamsd_analysis(trajectories, frames, params, show_plots, save_plots, run_analysis_verb, animated_plot_results):
    
    maxLagtime = 300*params['fps'] # maximum lagtime to be considered in the tamsd analysis, 300 seconds
    x_diffusive = np.linspace(50, maxLagtime/params['fps'], int((maxLagtime/params['fps'] + 1/params['fps'] - 50)*params['fps']))
    x_ballistic = np.linspace(1/params['fps'], 1, int((1-1/params['fps'])*params['fps'])+1)

    print("    Windowed TAMSD analysis...")
    if len(params['blue_particle_idx']) > 0:
        if run_analysis_verb:
            EMSD_wind_b, pw_exp_wind_b_ballistic, pw_exp_wind_b_diffusive = get_emsd_windowed_v2(params['n_windows'], params['startFrames'], params['endFrames'],
                                                              trajectories.loc[trajectories.particle.isin(params['blue_particle_idx']), ['particle', 'frame', 'x', 'y']],
                                                              params['pxDimension'], params['fps'], maxLagtime, x_ballistic, x_diffusive, params, progress_verb = True,
                                                              description = '    Computing blue droplets tamsd')
            EMSD_wind_b = np.transpose(EMSD_wind_b, (1, 0, 2))
            if os.path.isfile(f"./{params['analysis_data_path']}/tamsd_analysis/EMSD_windowed_b.npz"):
                os.remove(f"./{params['analysis_data_path']}/tamsd_analysis/EMSD_windowed_b.npz")
            np.savez(f"{params['analysis_data_path']}/tamsd_analysis/EMSD_windowed_b.npz", EMSD_wind_b = EMSD_wind_b, pw_exp_wind_b_ballistic = pw_exp_wind_b_ballistic, pw_exp_wind_b_diffusive = pw_exp_wind_b_diffusive)
        else:
            data = np.load(f"{params['analysis_data_path']}/tamsd_analysis/EMSD_windowed_b.npz")
            EMSD_wind_b = data['EMSD_wind_b']
            pw_exp_wind_b_ballistic = data['pw_exp_wind_b_ballistic']
            pw_exp_wind_b_diffusive = data['pw_exp_wind_b_diffusive']
    
    if len(params['red_particle_idx']) > 0:
        if run_analysis_verb:
            EMSD_wind_r, pw_exp_wind_r_ballistic, pw_exp_wind_r_diffusive = get_emsd_windowed_v2(params['n_windows'], params['startFrames'], params['endFrames'],
                                                              trajectories.loc[trajectories.particle.isin(params['red_particle_idx']), ['particle', 'frame', 'x', 'y']],
                                                              params['pxDimension'], params['fps'], maxLagtime, x_ballistic, x_diffusive, params, progress_verb = True,
                                                              description = '    Computing red droplets tamsd ')
            EMSD_wind_r = np.transpose(EMSD_wind_r, (1, 0, 2))
            if os.path.isfile(f"./{params['analysis_data_path']}/tamsd_analysis/EMSD_windowed_r.npz"):
                os.remove(f"./{params['analysis_data_path']}/tamsd_analysis/EMSD_windowed_r.npz")
            np.savez(f"{params['analysis_data_path']}/tamsd_analysis/EMSD_windowed_r.npz", EMSD_wind_r = EMSD_wind_r, pw_exp_wind_r_ballistic = pw_exp_wind_r_ballistic, pw_exp_wind_r_diffusive = pw_exp_wind_r_diffusive)
        else:
            data = np.load(f"{params['analysis_data_path']}/tamsd_analysis/EMSD_windowed_r.npz")
            EMSD_wind_r = data['EMSD_wind_r']
            pw_exp_wind_r_ballistic = data['pw_exp_wind_r_ballistic']
            pw_exp_wind_r_diffusive = data['pw_exp_wind_r_diffusive']
                
    if 1:
        gs = gridspec.GridSpec(2, 10)
        fig = plt.figure(figsize = (18, 6))
        i, step = 0, params['steps_plot'][0]
        ax1 = fig.add_subplot(gs[0, :2])
        if len(params['blue_particle_idx']) > 0:
            ax1.plot(np.arange(1, maxLagtime + 1, 1)/params['fps'], EMSD_wind_b[0, step], 'b-', label = 'Blue droplets')
            ax1.fill_between(np.arange(1, maxLagtime + 1, 1)/params['fps'], EMSD_wind_b[0, step] - 2/np.sqrt(len(params['blue_particle_idx'])) * EMSD_wind_b[1, step],\
                            EMSD_wind_b[0, step] + 2/np.sqrt(len(params['blue_particle_idx'])) * EMSD_wind_b[1, step], alpha=0.5, edgecolor='#00FFFF', facecolor='#F0FFFF')
        if len(params['red_particle_idx']) > 0:
            ax1.plot(np.arange(1, maxLagtime + 1, 1)/params['fps'], EMSD_wind_r[0, step], 'r-', label = 'Red droplets')
            ax1.fill_between(np.arange(1, maxLagtime + 1, 1)/params['fps'], EMSD_wind_r[0, step] - 2/np.sqrt(len(params['red_particle_idx'])) * EMSD_wind_r[1, step],\
                            EMSD_wind_r[0, step] + 2/np.sqrt(len(params['red_particle_idx'])) * EMSD_wind_r[1, step], alpha=0.5, edgecolor='#FF0000', facecolor='#FF5A52')
        ax1.grid(linewidth = 0.2)
        ax1.set(xscale = 'log', yscale = 'log', title = f"Stage {i + 1}", xlabel = r'$\tau$ [s]', ylabel = r'$\langle \Delta r^2 \rangle$ [$mm^2$]')
        ax1.legend(fontsize = 10)
        i, step = 1, params['steps_plot'][1]
        ax2 = fig.add_subplot(gs[0, 2:4], sharex = ax1, sharey = ax1)
        if len(params['blue_particle_idx']) > 0:
            ax2.plot(np.arange(1, maxLagtime + 1, 1)/params['fps'], EMSD_wind_b[0, step], 'b-', label = 'Blue droplets')
            ax2.fill_between(np.arange(1, maxLagtime + 1, 1)/params['fps'], EMSD_wind_b[0, step] - 2/np.sqrt(len(params['blue_particle_idx'])) * EMSD_wind_b[1, step],\
                            EMSD_wind_b[0, step] + 2/np.sqrt(len(params['blue_particle_idx'])) * EMSD_wind_b[1, step], alpha=0.5, edgecolor='#00FFFF', facecolor='#F0FFFF')
        if len(params['red_particle_idx']) > 0:
            ax2.plot(np.arange(1, maxLagtime + 1, 1)/params['fps'], EMSD_wind_r[0, step], 'r-', label = 'Red droplets')
            ax2.fill_between(np.arange(1, maxLagtime + 1, 1)/params['fps'], EMSD_wind_r[0, step] - 2/np.sqrt(len(params['red_particle_idx'])) * EMSD_wind_r[1, step],\
                            EMSD_wind_r[0, step] + 2/np.sqrt(len(params['red_particle_idx'])) * EMSD_wind_r[1, step], alpha=0.5, edgecolor='#FF0000', facecolor='#FF5A52')
        ax2.grid(linewidth = 0.2)
        ax2.set(xscale = 'log', yscale = 'log', title = f"Stage {i + 1}", xlabel = r'$\tau$ [s]')
        plt.setp(ax2.get_yticklabels(), visible=False)
        i, step = 2, params['steps_plot'][2]
        ax3 = fig.add_subplot(gs[0, 4:6], sharex = ax1, sharey = ax1)
        if len(params['blue_particle_idx']) > 0:
            ax3.plot(np.arange(1, maxLagtime + 1, 1)/params['fps'], EMSD_wind_b[0, step], 'b-', label = 'Blue droplets')
            ax3.fill_between(np.arange(1, maxLagtime + 1, 1)/params['fps'], EMSD_wind_b[0, step] - 2/np.sqrt(len(params['blue_particle_idx'])) * EMSD_wind_b[1, step],\
                            EMSD_wind_b[0, step] + 2/np.sqrt(len(params['blue_particle_idx'])) * EMSD_wind_b[1, step], alpha=0.5, edgecolor='#00FFFF', facecolor='#F0FFFF')
        if len(params['red_particle_idx']) > 0:
            ax3.plot(np.arange(1, maxLagtime + 1, 1)/params['fps'], EMSD_wind_r[0, step], 'r-', label = 'Red droplets')
            ax3.fill_between(np.arange(1, maxLagtime + 1, 1)/params['fps'], EMSD_wind_r[0, step] - 2/np.sqrt(len(params['red_particle_idx'])) * EMSD_wind_r[1, step],\
                            EMSD_wind_r[0, step] + 2/np.sqrt(len(params['red_particle_idx'])) * EMSD_wind_r[1, step], alpha=0.5, edgecolor='#FF0000', facecolor='#FF5A52')
        ax3.grid(linewidth = 0.2)
        ax3.set(xscale = 'log', yscale = 'log', title = f"Stage {i + 1}", xlabel = r'$\tau$ [s]')
        plt.setp(ax3.get_yticklabels(), visible=False)
        i, step = 3, params['steps_plot'][3]
        ax4 = fig.add_subplot(gs[0, 6:8], sharex = ax1, sharey = ax1)
        if len(params['blue_particle_idx']) > 0:
            ax4.plot(np.arange(1, maxLagtime + 1, 1)/params['fps'], EMSD_wind_b[0, step], 'b-', label = 'Blue droplets')
            ax4.fill_between(np.arange(1, maxLagtime + 1, 1)/params['fps'], EMSD_wind_b[0, step] - 2/np.sqrt(len(params['blue_particle_idx'])) * EMSD_wind_b[1, step],\
                            EMSD_wind_b[0, step] + 2/np.sqrt(len(params['blue_particle_idx'])) * EMSD_wind_b[1, step], alpha=0.5, edgecolor='#00FFFF', facecolor='#F0FFFF')
        if len(params['red_particle_idx']) > 0:
            ax4.plot(np.arange(1, maxLagtime + 1, 1)/params['fps'], EMSD_wind_r[0, step], 'r-', label = 'Red droplets')
            ax4.fill_between(np.arange(1, maxLagtime + 1, 1)/params['fps'], EMSD_wind_r[0, step] - 2/np.sqrt(len(params['red_particle_idx'])) * EMSD_wind_r[1, step],\
                            EMSD_wind_r[0, step] + 2/np.sqrt(len(params['red_particle_idx'])) * EMSD_wind_r[1, step], alpha=0.5, edgecolor='#FF0000', facecolor='#FF5A52')
        ax4.grid(linewidth = 0.2)
        ax4.set(xscale = 'log', yscale = 'log', title = f"Stage {i + 1}", xlabel = r'$\tau$ [s]')
        plt.setp(ax4.get_yticklabels(), visible=False)
        i, step = 4, params['steps_plot'][4]
        ax5 = fig.add_subplot(gs[0, 8:10], sharex = ax1, sharey = ax1)
        if len(params['blue_particle_idx']) > 0:
            ax5.plot(np.arange(1, maxLagtime + 1, 1)/params['fps'], EMSD_wind_b[0, step], 'b-', label = 'Blue droplets')
            ax5.fill_between(np.arange(1, maxLagtime + 1, 1)/params['fps'], EMSD_wind_b[0, step] - 2/np.sqrt(len(params['blue_particle_idx'])) * EMSD_wind_b[1, step],\
                            EMSD_wind_b[0, step] + 2/np.sqrt(len(params['blue_particle_idx'])) * EMSD_wind_b[1, step], alpha=0.5, edgecolor='#00FFFF', facecolor='#F0FFFF')
        if len(params['red_particle_idx']) > 0:
            ax5.plot(np.arange(1, maxLagtime + 1, 1)/params['fps'], EMSD_wind_r[0, step], 'r-', label = 'Red droplets')
            ax5.fill_between(np.arange(1, maxLagtime + 1, 1)/params['fps'], EMSD_wind_r[0, step] - 2/np.sqrt(len(params['red_particle_idx'])) * EMSD_wind_r[1, step],\
                            EMSD_wind_r[0, step] + 2/np.sqrt(len(params['red_particle_idx'])) * EMSD_wind_r[1, step], alpha=0.5, edgecolor='#FF0000', facecolor='#FF5A52')
        ax5.grid(linewidth = 0.2)
        ax5.set(xscale = 'log', yscale = 'log', title = f"Stage {i + 1}", xlabel = r'$\tau$ [s]')
        plt.setp(ax5.get_yticklabels(), visible=False)
        ax6 = fig.add_subplot(gs[1, :5])
        if len(params['blue_particle_idx']) > 0:
            ax6.plot(params['window_center_sec'], pw_exp_wind_b_diffusive[:, 0, 1], 'b-', label = 'Blue droplets')
        if len(params['red_particle_idx']) > 0:
            ax6.plot(params['window_center_sec'], pw_exp_wind_r_diffusive[:, 0, 1], 'r-', label = 'Red droplets ')
        ax6.plot(params['window_center_sec'], np.ones(params['n_windows']), 'k-')
        ax6.set(xlabel = r'$t_w$ [s]', ylabel = r'$\alpha$', ylim = (-0.1, 2.1))#, title = 'Scaling exponents')
        ax6.legend(loc = (0.09, 0.7), fontsize = 10)
        ax6.grid(linewidth = 0.2)
        for i, frame in enumerate(params['frames_stages']):
            ax6.bar(frame/params['fps'], height = 2000, width = params['window_length'], bottom = -10, color = params['stages_shades'][i], alpha = 0.5)
        ax7 = fig.add_subplot(gs[1, 5:], sharex = ax6)
        if len(params['blue_particle_idx']) > 0:
            ax7.plot(params['window_center_sec'], pw_exp_wind_b_diffusive[:, 0, 0], 'b-')
        if len(params['red_particle_idx']) > 0:
            ax7.plot(params['window_center_sec'], pw_exp_wind_r_diffusive[:, 0, 0], 'r-')
        for i, frame in enumerate(params['frames_stages']):
            ax7.bar(frame/params['fps'], height = 2000, width = params['window_length'], bottom = -10, color = params['stages_shades'][i], alpha = 0.5, label = f"Stage {i+1}")
        ax7.set(ylim=(-0.1, 20), xlim = (-200, params['max_window_sec']))
        ax7.legend(loc = (0.15, 0.4), fontsize = 10)
        ax7.set(xlabel = r'$t_w$ [s]', ylabel =r'$K{_\alpha} \; [mm^2/s^\alpha]$')#, title = 'Generalized diffusion coefficients')
        ax7.grid(linewidth = 0.2)
        ax1.text(0.0, 1.0, 'a)', transform=(ax1.transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)), fontsize='medium', va='bottom')
        ax2.text(0.0, 1.0, 'b)', transform=(ax2.transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)), fontsize='medium', va='bottom')
        ax3.text(0.0, 1.0, 'c)', transform=(ax3.transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)), fontsize='medium', va='bottom')
        ax4.text(0.0, 1.0, 'd)', transform=(ax4.transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)), fontsize='medium', va='bottom')
        ax5.text(0.0, 1.0, 'e)', transform=(ax5.transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)), fontsize='medium', va='bottom')
        ax6.text(0.0, 1.0, 'f)', transform=(ax6.transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)), fontsize='medium', va='bottom')
        ax7.text(0.0, 1.0, 'g)', transform=(ax7.transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)), fontsize='medium', va='bottom')
        #plt.suptitle(f"EMSD of system {params['system_name']}")
        plt.tight_layout(w_pad = 0.01)
        if save_plots:
            plt.savefig(f"./{params['res_path']}/tamsd_analysis/EMSD_wind_stages_{params['n_stages']}_v2.png", bbox_inches='tight')
            plt.savefig(f"./{params['pdf_res_path']}/tamsd_analysis/EMSD_wind_stages_{params['n_stages']}_v2.pdf", bbox_inches='tight')
        if show_plots:
            plt.show()
        else:
            plt.close()
            
        
        gs = gridspec.GridSpec(3, 10)
        fig = plt.figure(figsize = (18, 10))
        i, step = 0, params['steps_plot'][0]
        ax1 = fig.add_subplot(gs[0, :2])
        if len(params['blue_particle_idx']) > 0:
            ax1.plot(np.arange(1, maxLagtime + 1, 1)/params['fps'], EMSD_wind_b[0, step], 'b-', label = 'Blue droplets')
            ax1.fill_between(np.arange(1, maxLagtime + 1, 1)/params['fps'], EMSD_wind_b[0, step] - 2/np.sqrt(len(params['blue_particle_idx'])) * EMSD_wind_b[1, step],\
                            EMSD_wind_b[0, step] + 2/np.sqrt(len(params['blue_particle_idx'])) * EMSD_wind_b[1, step], alpha=0.5, edgecolor='#00FFFF', facecolor='#F0FFFF')
        if len(params['red_particle_idx']) > 0:
            ax1.plot(np.arange(1, maxLagtime + 1, 1)/params['fps'], EMSD_wind_r[0, step], 'r-', label = 'Red droplets')
            ax1.fill_between(np.arange(1, maxLagtime + 1, 1)/params['fps'], EMSD_wind_r[0, step] - 2/np.sqrt(len(params['red_particle_idx'])) * EMSD_wind_r[1, step],\
                            EMSD_wind_r[0, step] + 2/np.sqrt(len(params['red_particle_idx'])) * EMSD_wind_r[1, step], alpha=0.5, edgecolor='#FF0000', facecolor='#FF5A52')
        ax1.grid(linewidth = 0.2)
        ax1.set(xscale = 'log', yscale = 'log', title = f"Stage {i + 1}", xlabel = r'$\tau$ [s]', ylabel = r'$\langle \Delta r^2 \rangle$ [$mm^2$]')
        ax1.legend(fontsize = 10)
        i, step = 1, params['steps_plot'][1]
        ax2 = fig.add_subplot(gs[0, 2:4], sharex = ax1, sharey = ax1)
        if len(params['blue_particle_idx']) > 0:
            ax2.plot(np.arange(1, maxLagtime + 1, 1)/params['fps'], EMSD_wind_b[0, step], 'b-', label = 'Blue droplets')
            ax2.fill_between(np.arange(1, maxLagtime + 1, 1)/params['fps'], EMSD_wind_b[0, step] - 2/np.sqrt(len(params['blue_particle_idx'])) * EMSD_wind_b[1, step],\
                            EMSD_wind_b[0, step] + 2/np.sqrt(len(params['blue_particle_idx'])) * EMSD_wind_b[1, step], alpha=0.5, edgecolor='#00FFFF', facecolor='#F0FFFF')
        if len(params['red_particle_idx']) > 0:
            ax2.plot(np.arange(1, maxLagtime + 1, 1)/params['fps'], EMSD_wind_r[0, step], 'r-', label = 'Red droplets')
            ax2.fill_between(np.arange(1, maxLagtime + 1, 1)/params['fps'], EMSD_wind_r[0, step] - 2/np.sqrt(len(params['red_particle_idx'])) * EMSD_wind_r[1, step],\
                            EMSD_wind_r[0, step] + 2/np.sqrt(len(params['red_particle_idx'])) * EMSD_wind_r[1, step], alpha=0.5, edgecolor='#FF0000', facecolor='#FF5A52')
        ax2.grid(linewidth = 0.2)
        ax2.set(xscale = 'log', yscale = 'log', title = f"Stage {i + 1}", xlabel = r'$\tau$ [s]')
        plt.setp(ax2.get_yticklabels(), visible=False)
        i, step = 2, params['steps_plot'][2]
        ax3 = fig.add_subplot(gs[0, 4:6], sharex = ax1, sharey = ax1)
        if len(params['blue_particle_idx']) > 0:
            ax3.plot(np.arange(1, maxLagtime + 1, 1)/params['fps'], EMSD_wind_b[0, step], 'b-', label = 'Blue droplets')
            ax3.fill_between(np.arange(1, maxLagtime + 1, 1)/params['fps'], EMSD_wind_b[0, step] - 2/np.sqrt(len(params['blue_particle_idx'])) * EMSD_wind_b[1, step],\
                            EMSD_wind_b[0, step] + 2/np.sqrt(len(params['blue_particle_idx'])) * EMSD_wind_b[1, step], alpha=0.5, edgecolor='#00FFFF', facecolor='#F0FFFF')
        if len(params['red_particle_idx']) > 0:
            ax3.plot(np.arange(1, maxLagtime + 1, 1)/params['fps'], EMSD_wind_r[0, step], 'r-', label = 'Red droplets')
            ax3.fill_between(np.arange(1, maxLagtime + 1, 1)/params['fps'], EMSD_wind_r[0, step] - 2/np.sqrt(len(params['red_particle_idx'])) * EMSD_wind_r[1, step],\
                            EMSD_wind_r[0, step] + 2/np.sqrt(len(params['red_particle_idx'])) * EMSD_wind_r[1, step], alpha=0.5, edgecolor='#FF0000', facecolor='#FF5A52')
        ax3.grid(linewidth = 0.2)
        ax3.set(xscale = 'log', yscale = 'log', title = f"Stage {i + 1}", xlabel = r'$\tau$ [s]')
        plt.setp(ax3.get_yticklabels(), visible=False)
        i, step = 3, params['steps_plot'][3]
        ax4 = fig.add_subplot(gs[0, 6:8], sharex = ax1, sharey = ax1)
        if len(params['blue_particle_idx']) > 0:
            ax4.plot(np.arange(1, maxLagtime + 1, 1)/params['fps'], EMSD_wind_b[0, step], 'b-', label = 'Blue droplets')
            ax4.fill_between(np.arange(1, maxLagtime + 1, 1)/params['fps'], EMSD_wind_b[0, step] - 2/np.sqrt(len(params['blue_particle_idx'])) * EMSD_wind_b[1, step],\
                            EMSD_wind_b[0, step] + 2/np.sqrt(len(params['blue_particle_idx'])) * EMSD_wind_b[1, step], alpha=0.5, edgecolor='#00FFFF', facecolor='#F0FFFF')
        if len(params['red_particle_idx']) > 0:
            ax4.plot(np.arange(1, maxLagtime + 1, 1)/params['fps'], EMSD_wind_r[0, step], 'r-', label = 'Red droplets')
            ax4.fill_between(np.arange(1, maxLagtime + 1, 1)/params['fps'], EMSD_wind_r[0, step] - 2/np.sqrt(len(params['red_particle_idx'])) * EMSD_wind_r[1, step],\
                            EMSD_wind_r[0, step] + 2/np.sqrt(len(params['red_particle_idx'])) * EMSD_wind_r[1, step], alpha=0.5, edgecolor='#FF0000', facecolor='#FF5A52')
        ax4.grid(linewidth = 0.2)
        ax4.set(xscale = 'log', yscale = 'log', title = f"Stage {i + 1}", xlabel = r'$\tau$ [s]')
        plt.setp(ax4.get_yticklabels(), visible=False)
        i, step = 4, params['steps_plot'][4]
        ax5 = fig.add_subplot(gs[0, 8:10], sharex = ax1, sharey = ax1)
        if len(params['blue_particle_idx']) > 0:
            ax5.plot(np.arange(1, maxLagtime + 1, 1)/params['fps'], EMSD_wind_b[0, step], 'b-', label = 'Blue droplets')
            ax5.fill_between(np.arange(1, maxLagtime + 1, 1)/params['fps'], EMSD_wind_b[0, step] - 2/np.sqrt(len(params['blue_particle_idx'])) * EMSD_wind_b[1, step],\
                            EMSD_wind_b[0, step] + 2/np.sqrt(len(params['blue_particle_idx'])) * EMSD_wind_b[1, step], alpha=0.5, edgecolor='#00FFFF', facecolor='#F0FFFF')
        if len(params['red_particle_idx']) > 0:
            ax5.plot(np.arange(1, maxLagtime + 1, 1)/params['fps'], EMSD_wind_r[0, step], 'r-', label = 'Red droplets')
            ax5.fill_between(np.arange(1, maxLagtime + 1, 1)/params['fps'], EMSD_wind_r[0, step] - 2/np.sqrt(len(params['red_particle_idx'])) * EMSD_wind_r[1, step],\
                            EMSD_wind_r[0, step] + 2/np.sqrt(len(params['red_particle_idx'])) * EMSD_wind_r[1, step], alpha=0.5, edgecolor='#FF0000', facecolor='#FF5A52')
        ax5.grid(linewidth = 0.2)
        ax5.set(xscale = 'log', yscale = 'log', title = f"Stage {i + 1}", xlabel = r'$\tau$ [s]')
        plt.setp(ax5.get_yticklabels(), visible=False)

        ax6 = fig.add_subplot(gs[1, :5])
        if len(params['blue_particle_idx']) > 0:
            ax6.plot(params['window_center_sec'], pw_exp_wind_b_ballistic[:, 0, 1], 'b-', label = 'Blue droplets')
        if len(params['red_particle_idx']) > 0:
            ax6.plot(params['window_center_sec'], pw_exp_wind_r_ballistic[:, 0, 1], 'r-', label = 'Red droplets ')
        ax6.plot(params['window_center_sec'], np.ones(params['n_windows']), 'k-')
        ax6.set(xlabel = r'$t_w$ [s]', ylabel = r'$\alpha^{short}$', ylim = (-0.1, 2.1))#, title = 'Scaling exponents')
        #ax6.legend(loc = (0.09, 0.7), fontsize = 10)
        ax6.grid(linewidth = 0.2)
        for i, frame in enumerate(params['frames_stages']):
            ax6.bar(frame/params['fps'], height = 2000, width = params['window_length'], bottom = -10, color = params['stages_shades'][i], alpha = 0.5)
        ax7 = fig.add_subplot(gs[1, 5:], sharex = ax6)

        if len(params['blue_particle_idx']) > 0:
            ax7.plot(params['window_center_sec'], pw_exp_wind_b_ballistic[:, 0, 0], 'b-')
        if len(params['red_particle_idx']) > 0:
            ax7.plot(params['window_center_sec'], pw_exp_wind_r_ballistic[:, 0, 0], 'r-')
        for i, frame in enumerate(params['frames_stages']):
            ax7.bar(frame/params['fps'], height = 2000, width = params['window_length'], bottom = -10, color = params['stages_shades'][i], alpha = 0.5, label = f"Stage {i+1}")
        ax7.set(ylim=(-0.1, 5), xlim = (-200, params['max_window_sec']))
        #ax7.legend(loc = (0.15, 0.4), fontsize = 10)
        ax7.set(xlabel = r'$t_w$ [s]', ylabel =r'$K{_\alpha}^{short} \; [mm^2/s^\alpha]$')#, title = 'Generalized diffusion coefficients')
        ax7.grid(linewidth = 0.2)

        ax8 = fig.add_subplot(gs[2, :5])
        if len(params['blue_particle_idx']) > 0:
            ax8.plot(params['window_center_sec'], pw_exp_wind_b_diffusive[:, 0, 1], 'b-', label = 'Blue droplets')
        if len(params['red_particle_idx']) > 0:
            ax8.plot(params['window_center_sec'], pw_exp_wind_r_diffusive[:, 0, 1], 'r-', label = 'Red droplets ')
        ax8.plot(params['window_center_sec'], np.ones(params['n_windows']), 'k-')
        ax8.set(xlabel = r'$t_w$ [s]', ylabel = r'$\alpha^{long}$', ylim = (-0.1, 2.1))#, title = 'Scaling exponents')
        #ax8.legend(loc = (0.09, 0.7), fontsize = 10)
        ax8.grid(linewidth = 0.2)
        for i, frame in enumerate(params['frames_stages']):
            ax8.bar(frame/params['fps'], height = 2000, width = params['window_length'], bottom = -10, color = params['stages_shades'][i], alpha = 0.5)
        ax9 = fig.add_subplot(gs[2, 5:], sharex = ax8)
        if len(params['blue_particle_idx']) > 0:
            ax9.plot(params['window_center_sec'], pw_exp_wind_b_diffusive[:, 0, 0], 'b-')
        if len(params['red_particle_idx']) > 0:
            ax9.plot(params['window_center_sec'], pw_exp_wind_r_diffusive[:, 0, 0], 'r-')
        for i, frame in enumerate(params['frames_stages']):
            ax9.bar(frame/params['fps'], height = 2000, width = params['window_length'], bottom = -10, color = params['stages_shades'][i], alpha = 0.5, label = f"Stage {i+1}")
        ax9.set(ylim=(-0.1, 20), xlim = (-200, params['max_window_sec']))
        ax9.legend(loc = (0.1, 0.4), fontsize = 10)
        ax9.set(xlabel = r'$t_w$ [s]', ylabel =r'$K{_\alpha}^{long} \; [mm^2/s^\alpha]$')#, title = 'Generalized diffusion coefficients')
        ax9.grid(linewidth = 0.2)


        ax1.text(0.0, 1.0, 'a)', transform=(ax1.transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)), fontsize='medium', va='bottom')
        ax2.text(0.0, 1.0, 'b)', transform=(ax2.transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)), fontsize='medium', va='bottom')
        ax3.text(0.0, 1.0, 'c)', transform=(ax3.transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)), fontsize='medium', va='bottom')
        ax4.text(0.0, 1.0, 'd)', transform=(ax4.transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)), fontsize='medium', va='bottom')
        ax5.text(0.0, 1.0, 'e)', transform=(ax5.transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)), fontsize='medium', va='bottom')
        ax6.text(0.0, 1.0, 'f)', transform=(ax6.transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)), fontsize='medium', va='bottom')
        ax7.text(0.0, 1.0, 'g)', transform=(ax7.transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)), fontsize='medium', va='bottom')
        ax8.text(0.0, 1.0, 'h)', transform=(ax8.transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)), fontsize='medium', va='bottom')
        ax9.text(0.0, 1.0, 'i)', transform=(ax9.transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)), fontsize='medium', va='bottom')

        #plt.suptitle(f"EMSD of system {params['system_name']}")
        plt.tight_layout(w_pad = 0.01)
        if save_plots:
            plt.savefig(f"./{params['res_path']}/tamsd_analysis/EMSD_wind_stages_{params['n_stages']}_v3.png", bbox_inches='tight')
            plt.savefig(f"./{params['pdf_res_path']}/tamsd_analysis/EMSD_wind_stages_{params['n_stages']}_v3.pdf", bbox_inches='tight')
        if show_plots:
            plt.show()
        else:
            plt.close()

        if animated_plot_results:
            
            # Lower and Higher bounds for fill between 
            if len(params['blue_particle_idx']) > 0:
                Y1_msd_b = EMSD_wind_b[0] - 2/np.sqrt(len(params['blue_particle_idx'])) * EMSD_wind_b[1]
                Y2_msd_b = EMSD_wind_b[0] + 2/np.sqrt(len(params['blue_particle_idx'])) * EMSD_wind_b[1]
            
            if len(params['red_particle_idx']) > 0:
                Y1_msd_r = EMSD_wind_r[0] - 2/np.sqrt(len(params['red_particle_idx'])) * EMSD_wind_r[1]
                Y2_msd_r = EMSD_wind_r[0] + 2/np.sqrt(len(params['red_particle_idx'])) * EMSD_wind_r[1]

            gs = gridspec.GridSpec(2, 2)
            fig = plt.figure(figsize = (10, 5))
            ax = fig.add_subplot(gs[0, :])
            ax1 = fig.add_subplot(gs[1, 0])
            ax2 = fig.add_subplot(gs[1, 1])
            anim_running = True

            def update_plot(step):
                # update title
                title.set_text(f"EMSD of system {params['system_name']} at " + r'$t_w = $' + f"{params['startFrames'][step]/params['fps'] + params['window_length']/2} s")
                line.set_data(params['startFrames'][:step]/params['fps'] + params['window_length']/2, np.ones(step)) 
                ax1.set_xlim(frames[0]/params['fps'], params['startFrames'][step]/params['fps'] + params['window_length']/2 + 100)
                ax2.set_xlim(frames[0]/params['fps'], params['startFrames'][step]/params['fps'] + params['window_length']/2 + 100)
                
                # update MSD
                if len(params['blue_particle_idx']) > 0:
                    # update EMSD plot
                    msd_plot_b.set_ydata(EMSD_wind_b[0, step])
                    # update fill between
                    path = fill_graph_b.get_paths()[0]
                    verts = path.vertices
                    verts[1:maxLagtime+1, 1] = Y1_msd_b[step, :]
                    verts[maxLagtime+2:-1, 1] = Y2_msd_b[step, :][::-1]
                    # update power law exponents and generalized diffusion coefficients
                    line_b.set_data(params['startFrames'][:step]/params['fps'] + params['window_length']/2, pw_exp_wind_b_diffusive[:step, 0, 1])
                    line_b1.set_data(params['startFrames'][:step]/params['fps'] + params['window_length']/2, pw_exp_wind_b_diffusive[:step, 0, 0])
        
                if len(params['red_particle_idx']) > 0:
                    # update EMSD plot
                    msd_plot_r.set_ydata(EMSD_wind_r[0, step])
                    # update fill between
                    path = fill_graph_r.get_paths()[0]
                    verts = path.vertices
                    verts[1:maxLagtime+1, 1] = Y1_msd_r[step, :]
                    verts[maxLagtime+2:-1, 1] = Y2_msd_r[step, :][::-1]
                    # update power law exponents and generalized diffusion coefficients
                    line_r.set_data(params['startFrames'][:step]/params['fps'] + params['window_length']/2, pw_exp_wind_r_diffusive[:step, 0, 1]) 
                    line_r1.set_data(params['startFrames'][:step]/params['fps'] + params['window_length']/2, pw_exp_wind_r_diffusive[:step, 0, 0]) 

                
                
                if (len(params['blue_particle_idx']) > 0) & (len(params['red_particle_idx']) > 0):
                    return msd_plot_b, msd_plot_r, fill_graph_b, fill_graph_r, line, line_r, line_r1, line_b, line_b1,

                elif len(params['blue_particle_idx']) > 0 & len(params['red_particle_idx']) == 0:
                    return msd_plot_b, fill_graph_b, line, line_b, line_b1,

                elif len(params['blue_particle_idx']) == 0 & len(params['red_particle_idx']) > 0:
                    return msd_plot_r, fill_graph_r, line, line_r, line_r1,

            title = ax.set_title(f"EMSD of system {params['system_name']} at " + r'$t_w = $' + f"{params['startFrames'][0]/params['fps'] + params['window_length']/2} s")
            line, = ax1.plot(params['startFrames'][0]/params['fps'] + params['window_length']/2, 1, 'k-')
            
            if len(params['blue_particle_idx']) > 0:
                msd_plot_b = ax.plot(np.arange(1/params['fps'], maxLagtime/params['fps'] + 1/params['fps'], 1/params['fps']), EMSD_wind_b[0][0], 'b-', alpha=0.5, label = 'Blue droplets')[0]
                fill_graph_b = ax.fill_between(np.arange(1/params['fps'], maxLagtime/params['fps'] + 1/params['fps'], 1/params['fps']), Y1_msd_b[0], Y2_msd_b[0], alpha=0.5, edgecolor='#F0FFFF', facecolor='#00FFFF')
                line_b, = ax1.plot(params['startFrames'][0]/params['fps'] + params['window_length']/2, pw_exp_wind_b_diffusive[0, 0, 1], 'b-', alpha = 0.5, label = 'Blue droplets')	
                line_b1, = ax2.plot(params['startFrames'][0]/params['fps'] + params['window_length']/2, pw_exp_wind_b_diffusive[0, 0, 0], 'b-', alpha = 0.5, label = 'Blue droplets')
            
            if len(params['red_particle_idx']) > 0:
                msd_plot_r = ax.plot(np.arange(1/params['fps'], maxLagtime/params['fps'] + 1/params['fps'], 1/params['fps']), EMSD_wind_r[0][0], 'r-' , label = 'Red droplets')[0]
                fill_graph_r = ax.fill_between(np.arange(1/params['fps'], maxLagtime/params['fps'] + 1/params['fps'], 1/params['fps']), Y1_msd_r[0], Y2_msd_r[0], alpha=0.5, edgecolor='#FF5A52', facecolor='#FF5A52')
                line_r, = ax1.plot(params['startFrames'][0]/params['fps'] + params['window_length']/2, pw_exp_wind_r_diffusive[0, 0, 1], 'r-', alpha = 0.5, label = 'Red droplets')
                line_r1, = ax2.plot(params['startFrames'][0]/params['fps'] + params['window_length']/2, pw_exp_wind_r_diffusive[0, 0, 0], 'r-', alpha = 0.5, label = 'Red droplets')

            ax.set(xscale = 'log', yscale = 'log', ylabel = r'$\langle \Delta r^2 \rangle$ [$mm^2$]', xlabel = 'lag time $t$ [s]', ylim=(10**(-6), 10**(3)))
            ax2.set(xlabel = r'$t_w$ [s]', ylabel = r'$K{_\alpha} \; [mm^2/s^\alpha]$', ylim = (0, 20), title = 'Generalized diffusion coefficients')
            ax1.set(xlabel = r'$t_w$ [s]', ylabel = r'$\alpha$', ylim = (0, 2), title = 'Scaling exponents')

            ax.legend(fontsize = 10)
            ax.grid(linewidth = 0.2)
            ax1.grid(linewidth = 0.2)
            ax2.grid(linewidth = 0.2)
            ax1.set_xlim(frames[0]/params['fps'], params['startFrames'][0]/params['fps'] + params['window_length']/2 + 100)
            ax2.set_xlim(frames[0]/params['fps'], params['startFrames'][0]/params['fps'] + params['window_length']/2 + 100)

            plt.tight_layout()
            fig.canvas.mpl_connect('button_press_event', onClick)
            ani = FuncAnimation(fig, update_plot, params['n_windows'], interval = 5, blit=False)
            writer = FFMpegWriter(fps = 10, metadata = dict(artist='skandiz'), extra_args=['-vcodec', 'libx264'])
            if save_plots:
                ani.save(f"./{params['res_path']}/tamsd_analysis/EMSD_wind.mp4", writer=writer, dpi = 300)
            if show_plots:
                plt.show()
            else:
                plt.close()
            

    if (len(params['blue_particle_idx']) > 0) & (len(params['red_particle_idx']) > 0):
        return (EMSD_wind_b, EMSD_wind_r), (pw_exp_wind_b_ballistic, pw_exp_wind_r_ballistic), (pw_exp_wind_b_diffusive, pw_exp_wind_r_diffusive), maxLagtime
    elif (len(params['blue_particle_idx']) > 0) & (len(params['red_particle_idx']) == 0):
        return (EMSD_wind_b, None), (pw_exp_wind_b_ballistic, None), (pw_exp_wind_b_diffusive, None), maxLagtime
    elif (len(params['blue_particle_idx']) == 0) & (len(params['red_particle_idx']) > 0):
        return (None, EMSD_wind_r), (None, pw_exp_wind_r_ballistic), (None, pw_exp_wind_r_diffusive), maxLagtime