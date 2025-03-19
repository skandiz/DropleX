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
from matplotlib.animation import FuncAnimation
from matplotlib.transforms import ScaledTranslation

from analysis_utils import vacf_yupi_modified, vacf_windowed, onClick

def run_vacf_analysis(trajectories, params, show_plots, save_plots, run_analysis_verb, animated_plot_results):
    maxLagtime = 90 * params['fps']
    lag_times = np.arange(0, maxLagtime/params['fps'], 1/params['fps'])


    print('    Global Velocity autocovariance analysis...')
    
    if run_analysis_verb:
        if len(params['blue_particle_idx']) > 0:
            vacf_b, vacf_std_b = vacf_yupi_modified(trajectories.loc[trajectories.particle.isin(params['blue_particle_idx'])], params['fps'], params['pxDimension'], maxLagtime)
        if len(params['red_particle_idx']) > 0:
            vacf_r, vacf_std_r = vacf_yupi_modified(trajectories.loc[trajectories.particle.isin(params['red_particle_idx'])], params['fps'], params['pxDimension'], maxLagtime)
        if os.path.isfile(f"./{params['analysis_data_path']}/vacf_analysis/vacf_global.npz"):
            os.remove(f"./{params['analysis_data_path']}/vacf_analysis/vacf_global.npz")
        np.savez(f"./{params['analysis_data_path']}/vacf_analysis/vacf_global.npz", vacf_b = vacf_b, vacf_std_b = vacf_std_b, vacf_r = vacf_r, vacf_std_r = vacf_std_r)

    else:
        data = np.load(f"./{params['analysis_data_path']}/vacf_analysis/vacf_global.npz")
        vacf_b = data['vacf_b']
        vacf_std_b = data['vacf_std_b']
        vacf_r = data['vacf_r']
        vacf_std_r = data['vacf_std_r']
    
    
    if 1:
        fig, (ax, ax1) = plt.subplots(1, 2, figsize = (12, 4), sharex = True, sharey = True)
        if len(params['blue_particle_idx']) > 0:
            ax.errorbar(lag_times, vacf_b, fmt='o', markersize = 1, color = 'blue', label = 'Blue droplets')
            ax.fill_between(lag_times, vacf_b + 2/np.sqrt(len(params['blue_particle_idx'])) * vacf_std_b, vacf_b - 2/np.sqrt(len(params['blue_particle_idx'])) * vacf_std_b, alpha=1, edgecolor='#F0FFFf', facecolor='#00FFFf')
        ax.grid(linewidth = 0.2)
        ax.legend(fontsize = 10)
        ax.set(xlabel = 'Lag time [s]', ylabel = r'VACF [$(mm/s)^2$]')
        
        if len(params['red_particle_idx']) > 0:
            ax1.errorbar(lag_times, vacf_r, fmt='o', markersize = 1, color = 'red', label = 'Red droplets')
            ax1.fill_between(lag_times, vacf_r + 2/np.sqrt(len(params['red_particle_idx'])) * vacf_std_r, vacf_r -2/np.sqrt(len(params['red_particle_idx'])) *  vacf_std_r, alpha=1, edgecolor='#FF0000', facecolor='#FFCCCB')
        ax1.set(xlabel = 'Lag time [s]')
        ax1.grid(linewidth = 0.2)
        ax1.legend(fontsize = 10)
        ax.text(0.0, 1.0, 'a)', transform=(ax.transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)), fontsize='medium', va='bottom')
        ax1.text(0.0, 1.0, 'b)', transform=(ax1.transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)), fontsize='medium', va='bottom')
        plt.suptitle(f"Velocity autocorrelation function of system {params['system_name']}")
        plt.tight_layout()
        if save_plots: 
            plt.savefig(f"./{params['res_path']}/vacf_analysis/vacf.png", bbox_inches='tight')
            plt.savefig(f"./{params['pdf_res_path']}/vacf_analysis/vacf.pdf", bbox_inches='tight')
        if show_plots: 
            plt.show()
        else:
            plt.close()
        

    print('    Windowed velocity autocovariance analysis...')
    if len(params['blue_particle_idx']) > 0:
        if run_analysis_verb:
            vacf_wind_b, vacf_std_wind_b = vacf_windowed(trajectories.loc[trajectories.particle.isin(params['blue_particle_idx'])], params['n_windows'], params['startFrames'], params['endFrames'], params['fps'], params['pxDimension'], maxLagtime, progress_verb = True)
            if os.path.isfile(f"./{params['analysis_data_path']}/vacf_analysis/vacf_wind_b.npz"):
                os.remove(f"./{params['analysis_data_path']}/vacf_analysis/vacf_wind_b.npz")
            np.savez(f"./{params['analysis_data_path']}/vacf_analysis/vacf_wind_b.npz", vacf_wind_b = vacf_wind_b, vacf_std_wind_b = vacf_std_wind_b)
        else:
            data = np.load(f"./{params['analysis_data_path']}/vacf_analysis/vacf_wind_b.npz")
            vacf_wind_b = data['vacf_wind_b']
            vacf_std_wind_b = data['vacf_std_wind_b']
            
        vacf_wind_b = vacf_wind_b / vacf_wind_b[:, 0, np.newaxis]
        vacf_std_wind_b = np.sqrt((vacf_std_wind_b / vacf_wind_b[:, 0, np.newaxis])**2 + (vacf_wind_b * vacf_std_wind_b[:, 0, np.newaxis] / vacf_wind_b[:, 0, np.newaxis]**2)**2)

            
    if len(params['red_particle_idx']) > 0:
        if run_analysis_verb:
            vacf_wind_r, vacf_std_wind_r = vacf_windowed(trajectories.loc[trajectories.particle.isin(params['red_particle_idx'])], params['n_windows'], params['startFrames'], params['endFrames'], params['fps'], params['pxDimension'], maxLagtime, progress_verb = True)
            if os.path.isfile(f"./{params['analysis_data_path']}/vacf_analysis/vacf_wind_r.npz"):
                os.remove(f"./{params['analysis_data_path']}/vacf_analysis/vacf_wind_r.npz")
            np.savez(f"./{params['analysis_data_path']}/vacf_analysis/vacf_wind_r.npz", vacf_wind_r = vacf_wind_r, vacf_std_wind_r = vacf_std_wind_r)
        else:
            data = np.load(f"./{params['analysis_data_path']}/vacf_analysis/vacf_wind_r.npz")
            vacf_wind_r = data['vacf_wind_r']
            vacf_std_wind_r = data['vacf_std_wind_r']
        vacf_wind_r = vacf_wind_r / vacf_wind_r[:, 0, np.newaxis]
        vacf_std_wind_r = np.sqrt((vacf_std_wind_r / vacf_wind_r[:, 0, np.newaxis])**2 + (vacf_wind_r * vacf_std_wind_r[:, 0, np.newaxis] / vacf_wind_r[:, 0, np.newaxis]**2)**2)
        
        
    if 1:
        fig, ax = plt.subplots(1, 1, figsize = (8, 4))
        if len(params['blue_particle_idx']) > 0:
            ax.plot(params['window_center_sec'], 2/np.sqrt(len(params['blue_particle_idx'])) * vacf_std_wind_b[:, 0], 'b')
        if len(params['red_particle_idx']) > 0:
            ax.plot(params['window_center_sec'], 2/np.sqrt(len(params['red_particle_idx'])) * vacf_std_wind_r[:, 0], 'r')
        for i, step in enumerate(params['steps_plot']):
            ax.bar(params['frames_stages'][i]/params['fps'], height = 2000, width = params['window_length'], bottom = -10, color = params['stages_shades'][i], alpha = 0.5)
        ax.set(xlabel = 'Window time [s]', ylabel = r'$\sigma$', title = f"VACF standard deviation of system {params['system_name']}")
        if params['video_selection'] in ['25b25r_lowconc_1', '25b25r_lowconc_2', '25b25r_lowconc_3', '25b25r_lowconc_5', '25b25r_lowconc_6']:
            ax.set(xlim = (-200, 14000),  ylim = (-0.1, 3.5))
        elif params['video_selection'] in ['25b25r-1', '25b25r-2']:
            ax.set(ylim = (-0.1, 1.9))
        ax.legend(['Blue droplets', 'Red droplets'], fontsize = 10)
        ax.grid(linewidth = 0.2)
        if save_plots:
            plt.savefig(f"./{params['res_path']}/vacf_analysis/vacf_std_wind.png", bbox_inches='tight')
            plt.savefig(f"./{params['pdf_res_path']}/vacf_analysis/vacf_std_wind.pdf", bbox_inches='tight')
        if show_plots:
            plt.show()
        else:
            plt.close()

        fig, axs = plt.subplots(1, 5, figsize=(15, 4), sharex = True, sharey = True)
        for i, step in enumerate(params['steps_plot']):
            if len(params['blue_particle_idx']) > 0:
                axs[i].plot(lag_times, vacf_wind_b[step, :], 'b', label = 'Blue droplets')
                axs[i].fill_between(lag_times, vacf_wind_b[step, :] - 2/np.sqrt(len(params['blue_particle_idx'])) * vacf_std_wind_b[step, :], 
                                    vacf_wind_b[step, :] + 2/np.sqrt(len(params['blue_particle_idx'])) * vacf_std_wind_b[step, :], color = 'b', alpha = 0.2)
            if len(params['red_particle_idx']) > 0:
                axs[i].plot(lag_times, vacf_wind_r[step, :], 'r', label = 'Red droplets')
                axs[i].fill_between(lag_times, vacf_wind_r[step, :] - 2/np.sqrt(len(params['red_particle_idx'])) * vacf_std_wind_r[step, :], 
                                    vacf_wind_r[step, :] + 2/np.sqrt(len(params['red_particle_idx'])) * vacf_std_wind_r[step, :], color = 'r', alpha = 0.2)
            axs[i].set(xlabel = 'Lag time [s]', title = f"Stage {i+1}")
            axs[i].grid(linewidth = 0.2)
            axs[i].text(0.0, 1.0, f"{params['letter_labels'][i]}", transform=(axs[i].transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)), fontsize='medium', va='bottom')
        axs[0].set(ylabel = r'VACF [$(mm/s)^2$]', ylim = (-1.5, 1.5))
        if params['video_selection'] in ['25b25r_lowconc_1', '25b25r_lowconc_2', '25b25r_lowconc_3', '25b25r_lowconc_5', '25b25r_lowconc_6']:
            axs[0].set(ylim = (-0.5, 1.5), xlim = (-1, 50))
        plt.suptitle(f"Velocity autocorrelation function of system {params['system_name']}")
        plt.tight_layout()
        if save_plots:
            plt.savefig(f"./{params['res_path']}/vacf_analysis/vacf_wind_stages_5.png", bbox_inches='tight')
            plt.savefig(f"./{params['pdf_res_path']}/vacf_analysis/vacf_wind_stages_5.pdf", bbox_inches='tight')
        if show_plots:
            plt.show()
        else:
            plt.close()
        

        # Animated Plots
        if animated_plot_results:                
            if len(params['blue_particle_idx']) > 0:
                Y1 = vacf_wind_b - 2/np.sqrt(len(params['blue_particle_idx'])) * vacf_std_wind_b
                Y2 = vacf_wind_b + 2/np.sqrt(len(params['blue_particle_idx'])) * vacf_std_wind_b
                
            if len(params['red_particle_idx']) > 0:
                Y3 = vacf_wind_r - 2/np.sqrt(len(params['red_particle_idx'])) * vacf_std_wind_r
                Y4 = vacf_wind_r + 2/np.sqrt(len(params['red_particle_idx'])) * vacf_std_wind_r

            fig, (ax, ax1) = plt.subplots(2, 1, figsize = (8, 5), sharex = True, sharey = True)        
            anim_running = True   
                
            def update_plot(step):
                title.set_text(f"VACF of system {params['system_name']} at  " + r'$T_w$'  f"= {params['startFrames'][step]/params['fps'] + params['window_length']/2} s")
                
                if len(params['blue_particle_idx']) > 0:
                    line_b.set_ydata(vacf_wind_b[step, :]/vacf_wind_b[step, 0])
                    # update fill between
                    path = fill_b.get_paths()[0]
                    verts = path.vertices
                    verts[1:maxLagtime+1, 1] = Y1[step, :]
                    verts[maxLagtime+2:-1, 1] = Y2[step, :][::-1]
                    
                if len(params['red_particle_idx']) > 0:
                    line_r.set_ydata(vacf_wind_r[step, :]/vacf_wind_r[step, 0])
                    path = fill_r.get_paths()[0]
                    verts = path.vertices
                    verts[1:maxLagtime+1, 1] = Y3[step, :]
                    verts[maxLagtime+2:-1, 1] = Y4[step, :][::-1]
                
                return 0
            

            title = ax.set_title(f"VACF of system {params['system_name']} at  " + r'$T_w$'  f"= {params['startFrames'][0]/params['fps'] + params['window_length']/2} s")
            if len(params['blue_particle_idx']) > 0:
                line_b, = ax.plot(lag_times, vacf_wind_b[0, :]/vacf_wind_b[0, 0], 'b-', label = 'Blue droplets')
                fill_b = ax.fill_between(lag_times, Y2[0], Y1[0], alpha = 0.5, edgecolor = 'b', facecolor = 'b')
            ax.set(ylabel = r'vacf [$(mm/s)^2$]', xlabel = 'lag time $t$ [s]', ylim = (-1.5, 1.5))
            ax.grid(linewidth = 0.2)
            ax.legend(fontsize = 10)

            if len(params['red_particle_idx']) > 0:
                line_r, = ax1.plot(lag_times, vacf_wind_r[0, :]/vacf_wind_r[0, 0], 'r-', label = 'Red droplets')
                fill_r = ax1.fill_between(lag_times, Y4[0], Y3[0], alpha = 0.5, edgecolor = 'r', facecolor = 'r')
            ax1.set(ylabel = r'vacf [$(mm/s)^2$]', xlabel = 'lag time $t$ [s]', ylim = (-1.5, 1.5))
            ax1.grid(linewidth = 0.2)
            ax1.legend(fontsize = 10)
            plt.tight_layout()
            fig.canvas.mpl_connect('button_press_event', onClick)
            ani = FuncAnimation(fig, update_plot, params['n_windows'], interval = 5, blit=False)
            if save_plots: 
                ani.save(f"./{params['res_path']}/vacf_analysis/vacf_wind.mp4", fps = 30, extra_args=['-vcodec', 'libx264'])
            if show_plots:
                plt.show()
            else:
                plt.close()