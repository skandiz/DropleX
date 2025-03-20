import os
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({
	'font.size': 12,               # General font size
	'axes.titlesize': 14,          # Title font size
	'axes.labelsize': 12,          # Axishow_plotsnt size
	'legend.fontsize': 10,         # Legend font size
	'xtick.labelsize': 10,
	'ytick.labelsize': 10})
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.transforms import ScaledTranslation
from scipy.spatial import KDTree

from analysis_utils import compute_windowed_dimer_distribution, onClick

def run_dimer_analysis(trajectories, radii, params, show_plots, save_plots, run_analysis_verb, animated_plot_results):
    rDisk = (params['xmax']-params['xmin'])/2 * params['resolution']/(params['xmax']-params['xmin']) 
    r_bins = np.linspace(-rDisk, rDisk, 100)*params['pxDimension']
    dr = r_bins[1] - r_bins[0]
    print('    Windowed dimer distribution analysis...')
    if len(params['blue_particle_idx']) > 0:
        if run_analysis_verb:
            coords_blue = trajectories.loc[trajectories.particle.isin(params['blue_particle_idx']), ['x', 'y']].values.reshape(len(trajectories.frame.unique()), -1, 2)
            dimer_distr_windowed_bb = compute_windowed_dimer_distribution(coords_blue, coords_blue, r_bins, params['pxDimension'],
                                                                          True, params['n_windows'], params['startFrames'],
                                                                          params['endFrames'], description = '    Computing windowed dimer distribution for blue-blue dimers')
            if os.path.isfile(f"./{params['analysis_data_path']}/dimer_analysis/dimer_distr_windowed_bb.npz"):
                os.remove(f"./{params['analysis_data_path']}/dimer_analysis/dimer_distr_windowed_bb.npz")		
            np.savez(f"./{params['analysis_data_path']}/dimer_analysis/dimer_distr_windowed_bb.npz", dimer_distr_windowed_bb = dimer_distr_windowed_bb)
        else:
            data = np.load(f"./{params['analysis_data_path']}/dimer_analysis/dimer_distr_windowed_bb.npz")
            dimer_distr_windowed_bb = data['dimer_distr_windowed_bb']
            
    if len(params['red_particle_idx']) > 0:
        if run_analysis_verb:
            coords_red = trajectories.loc[trajectories.particle.isin(params['red_particle_idx']), ['x', 'y']].values.reshape(len(trajectories.frame.unique()), -1, 2)
            dimer_distr_windowed_rr = compute_windowed_dimer_distribution(coords_red, coords_red, r_bins, params['pxDimension'],
                                                                          True, params['n_windows'], params['startFrames'],
                                                                          params['endFrames'], description = '    Computing windowed dimer distribution for red-red dimers  ')
            if os.path.isfile(f"./{params['analysis_data_path']}/dimer_analysis/dimer_distr_windowed_rr.npz"):
                os.remove(f"./{params['analysis_data_path']}/dimer_analysis/dimer_distr_windowed_rr.npz")
            np.savez(f"./{params['analysis_data_path']}/dimer_analysis/dimer_distr_windowed_rr.npz", dimer_distr_windowed_rr = dimer_distr_windowed_rr)
        else:
            data = np.load(f"./{params['analysis_data_path']}/dimer_analysis/dimer_distr_windowed_rr.npz")
            dimer_distr_windowed_rr = data['dimer_distr_windowed_rr']

    if (len(params['blue_particle_idx']) > 0) & (len(params['red_particle_idx']) > 0):
        if run_analysis_verb:
            dimer_distr_windowed_br = compute_windowed_dimer_distribution(coords_blue, coords_red, r_bins, params['pxDimension'],
                                                                          False, params['n_windows'], params['startFrames'],
                                                                          params['endFrames'], description = '    Computing windowed dimer distribution for blue-red dimers ')
            dimer_distr_windowed_rb = compute_windowed_dimer_distribution(coords_red, coords_blue, r_bins, params['pxDimension'],
                                                                          False, params['n_windows'], params['startFrames'],
                                                                          params['endFrames'], description = '    Computing windowed dimer distribution for red-blue dimers ')
            if (os.path.isfile(f"./{params['analysis_data_path']}/dimer_analysis/dimer_distr_windowed_br.npz")) & (os.path.isfile(f"./{params['analysis_data_path']}/dimer_analysis/dimer_distr_windowed_rb.npz")): 
                os.remove(f"./{params['analysis_data_path']}/dimer_analysis/dimer_distr_windowed_br.npz")
                os.remove(f"./{params['analysis_data_path']}/dimer_analysis/dimer_distr_windowed_rb.npz")
            np.savez(f"./{params['analysis_data_path']}/dimer_analysis/dimer_distr_windowed_br.npz", dimer_distr_windowed_br = dimer_distr_windowed_br)
            np.savez(f"./{params['analysis_data_path']}/dimer_analysis/dimer_distr_windowed_rb.npz", dimer_distr_windowed_rb = dimer_distr_windowed_rb)
        else:
            data = np.load(f"./{params['analysis_data_path']}/dimer_analysis/dimer_distr_windowed_br.npz")
            dimer_distr_windowed_br = data['dimer_distr_windowed_br']
            data = np.load(f"./{params['analysis_data_path']}/dimer_analysis/dimer_distr_windowed_rb.npz")
            dimer_distr_windowed_rb = data['dimer_distr_windowed_rb']
        
    mean_wind_radius = np.zeros(params['n_windows'])
    for i in range(params['n_windows']):
        mean_wind_radius[i] = np.mean(radii[params['startFrames'][i]:params['endFrames'][i]])*params['pxDimension']
            
    if 1:
        v_max = np.max(np.concatenate((dimer_distr_windowed_bb, dimer_distr_windowed_rr, dimer_distr_windowed_br, dimer_distr_windowed_rb)))/5
        fig, axs = plt.subplots(4, params['n_stages'], figsize = (18, 14), sharex=True, sharey=True)
        for i, step in enumerate(params['steps_plot']):
            img = axs[0, i].imshow(dimer_distr_windowed_bb[step], extent=[r_bins[0], r_bins[-1], r_bins[0], r_bins[-1]], vmin = 0, vmax = v_max, cmap = 'Blues')
            axs[0, i].set(title = r'$D_{bb}$' + f"  at stage {i+1}")
            axs[1, i].imshow(dimer_distr_windowed_rr[step], extent=[r_bins[0], r_bins[-1], r_bins[0], r_bins[-1]], vmin = 0, vmax = v_max, cmap = 'Blues')
            axs[1, i].set(title = r'$D_{rr}$' + f"  at stage {i+1}")
            axs[2, i].imshow(dimer_distr_windowed_br[step], extent=[r_bins[0], r_bins[-1], r_bins[0], r_bins[-1]], vmin = 0, vmax = v_max, cmap = 'Blues')
            axs[2, i].set(title = r'$D_{br}$' + f"  at stage {i+1}")
            axs[3, i].imshow(dimer_distr_windowed_rb[step], extent=[r_bins[0], r_bins[-1], r_bins[0], r_bins[-1]], vmin = 0, vmax = v_max, cmap = 'Blues')
            axs[3, i].set(title = r'$D_{rb}$' + f"  at stage {i+1}")
            axs[3, i].set(xlabel = 'x [mm]')

        for i, ax in enumerate(axs.flatten()):
            ax.text(0.0, 1.0, f"{params['letter_labels'][i]}", transform=(ax.transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)), fontsize='medium', va='bottom')
        axs[0, 0].set(xticks = [- 20, - 10, 0, 10, 20], yticks = [- 20, - 10, 0, 10, 20])
        ax.set(xlim = (-20, 20), ylim = (-20, 20), xlabel = 'x [mm]', ylabel = 'y [mm]')
        axs[0, 0].set(ylabel = 'y [mm]')
        axs[1, 0].set(ylabel = 'y [mm]')
        axs[2, 0].set(ylabel = 'y [mm]')
        axs[3, 0].set(ylabel = 'y [mm]')
        plt.tight_layout()
        if save_plots:
            plt.savefig(f"{params['res_path']}/dimer_analysis/dimer_distributions_zoom.png", bbox_inches='tight')
            plt.savefig(f"{params['pdf_res_path']}/dimer_analysis/dimer_distributions_zoom.pdf", bbox_inches='tight')
        if show_plots:
            plt.show()
        else:
            plt.close()

        fig, axs = plt.subplots(1, 2, figsize = (10, 4), sharex = True, sharey = True)
        for angle in [0, 60, 120, 180, 240, 300]:
            x = 4.2 * mean_wind_radius[params['steps_plot'][4]] * np.cos(np.radians(angle))
            y = 4.2 * mean_wind_radius[params['steps_plot'][4]] * np.sin(np.radians(angle))
            for j in range(2):
                axs[j].plot([0, x], [0, y], color='white', linestyle='--')
                if angle == 60:
                    axs[j].text(x, y + 1, f"{angle}°", fontsize = 10, color = 'white', ha = 'left', va = 'bottom')
                elif angle == 120:
                    axs[j].text(x, y + 1, f"{angle}°", fontsize = 10, color = 'white', ha = 'right', va = 'bottom')
                elif angle == 180:
                    axs[j].text(x - 1, y, f"{angle}°", fontsize = 10, color = 'white', ha = 'right', va = 'center')
                elif angle == 240:
                    axs[j].text(x, y - 1, f"{angle}°", fontsize = 10, color = 'white', ha = 'right', va = 'top')
                elif angle == 300:
                    axs[j].text(x, y - 1, f"{angle}°", fontsize = 10, color = 'white', ha = 'left', va = 'top')

        img = axs[0].imshow(dimer_distr_windowed_bb[params['steps_plot'][1]], extent = [r_bins[0], r_bins[-1], r_bins[0], r_bins[-1]], vmin = 0, vmax = v_max, cmap = 'gnuplot')
        axs[0].set(title = r'$D_{bb}$' + f"  at stage 2")
        img = axs[1].imshow(dimer_distr_windowed_rr[params['steps_plot'][1]], extent = [r_bins[0], r_bins[-1], r_bins[0], r_bins[-1]], vmin = 0, vmax = v_max, cmap = 'gnuplot')
        axs[1].set(title = r'$D_{rr}$' + f"  at stage 2")

        axs[0].add_artist(plt.Circle((0, 0), 2.2*mean_wind_radius[params['steps_plot'][1]], fill = False, edgecolor = 'white', linewidth = 1))
        axs[0].add_artist(plt.Circle((0, 0), 4.2*mean_wind_radius[params['steps_plot'][1]], fill = False, edgecolor = 'white', linewidth = 1))
        axs[1].add_artist(plt.Circle((0, 0), 2.2*mean_wind_radius[params['steps_plot'][1]], fill = False, edgecolor = 'white', linewidth = 1))
        axs[1].add_artist(plt.Circle((0, 0), 4.2*mean_wind_radius[params['steps_plot'][1]], fill = False, edgecolor = 'white', linewidth = 1))

        axs[0].plot(2*x, np.zeros_like(2*x), 'w--', lw = 1)
        axs[0].text(2*x, 1, r'$\sim 4 d$', fontsize = 10, color = 'white', ha = 'left', va = 'bottom')
        axs[0].text(x, 1, r'$\sim 2 d$', fontsize = 10, color = 'white', ha = 'left', va = 'bottom')

        axs[0].set(xlim = (-10, 10), ylim = (-10, 10), xlabel = 'x [mm]', ylabel = 'y [mm]')
        axs[0].set(xticks = [- 10, - 5, 0, 5, 10], yticks = [- 10, - 5, 0, 5, 10])
        fig.colorbar(img, ax = axs.ravel().tolist())
        if save_plots:
            plt.savefig(f"./{params['res_path']}/dimer_analysis/dimer_distr_windowed_zoom_stage_2.png", bbox_inches='tight')
            plt.savefig(f"./{params['pdf_res_path']}/dimer_analysis/dimer_distr_windowed_zoom_stage_2.pdf", bbox_inches='tight')
        if show_plots:
            plt.show()
        else:
            plt.close()
            
        fig, axs = plt.subplots(1, 2, figsize = (10, 4), sharex = True, sharey = True)
        for angle in [0, 60, 120, 180, 240, 300]:
            x = 4.2 * mean_wind_radius[params['steps_plot'][4]] * np.cos(np.radians(angle))
            y = 4.2 * mean_wind_radius[params['steps_plot'][4]] * np.sin(np.radians(angle))
            for j in range(2):
                axs[j].plot([0, x], [0, y], color='white', linestyle='--')
                if angle == 60:
                    axs[j].text(x, y + 1, f"{angle}°", fontsize = 10, color = 'white', ha = 'left', va = 'bottom')
                elif angle == 120:
                    axs[j].text(x, y + 1, f"{angle}°", fontsize = 10, color = 'white', ha = 'right', va = 'bottom')
                elif angle == 180:
                    axs[j].text(x - 1, y, f"{angle}°", fontsize = 10, color = 'white', ha = 'right', va = 'center')
                elif angle == 240:
                    axs[j].text(x, y - 1, f"{angle}°", fontsize = 10, color = 'white', ha = 'right', va = 'top')
                elif angle == 300:
                    axs[j].text(x, y - 1, f"{angle}°", fontsize = 10, color = 'white', ha = 'left', va = 'top')

        img = axs[0].imshow(dimer_distr_windowed_bb[params['steps_plot'][4]], extent = [r_bins[0], r_bins[-1], r_bins[0], r_bins[-1]], vmin = 0, vmax = v_max, cmap = 'gnuplot')
        axs[0].set(title = r'$D_{bb}$' + f"  at stage 5")
        img = axs[1].imshow(dimer_distr_windowed_rr[params['steps_plot'][4]], extent = [r_bins[0], r_bins[-1], r_bins[0], r_bins[-1]], vmin = 0, vmax = v_max, cmap = 'gnuplot')
        axs[1].set(title = r'$D_{rr}$' + f"  at stage 5")


        axs[0].add_artist(plt.Circle((0, 0), 2.2*mean_wind_radius[params['steps_plot'][4]], fill = False, edgecolor = 'white', linewidth = 1))
        axs[0].add_artist(plt.Circle((0, 0), 4.2*mean_wind_radius[params['steps_plot'][4]], fill = False, edgecolor = 'white', linewidth = 1))
        axs[1].add_artist(plt.Circle((0, 0), 2.2*mean_wind_radius[params['steps_plot'][4]], fill = False, edgecolor = 'white', linewidth = 1))
        axs[1].add_artist(plt.Circle((0, 0), 4.2*mean_wind_radius[params['steps_plot'][4]], fill = False, edgecolor = 'white', linewidth = 1))

        axs[0].plot(2*x, np.zeros_like(2*x), 'w--', lw = 1)
        axs[0].text(2*x, 1, r'$\sim 4 d$', fontsize = 10, color = 'white', ha = 'left', va = 'bottom')
        axs[0].text(x, 1, r'$\sim 2 d$', fontsize = 10, color = 'white', ha = 'left', va = 'bottom')

        axs[0].set(xlim = (-10, 10), ylim = (-10, 10), xlabel = 'x [mm]', ylabel = 'y [mm]')
        axs[0].set(xticks = [- 10, - 5, 0, 5, 10], yticks = [- 10, - 5, 0, 5, 10])
        fig.colorbar(img, ax = axs.ravel().tolist())
        if save_plots:
            plt.savefig(f"./{params['res_path']}/dimer_analysis/dimer_distr_windowed_zoom_stage_5.png", bbox_inches='tight')
            plt.savefig(f"./{params['pdf_res_path']}/dimer_analysis/dimer_distr_windowed_zoom_stage_5.pdf", bbox_inches='tight')
        if show_plots:
            plt.show()
        else:
            plt.close()
                        
        if (len(params['blue_particle_idx']) == 0) or (len(params['red_particle_idx']) == 0):
            fig, axs = plt.subplots(1, params['n_stages'], figsize = (25, 5), sharex=True, sharey=True)
            for i, step in enumerate(params['steps_plot']):
                if (len(params['blue_particle_idx']) > 0):
                    v_max = np.max(np.concatenate((dimer_distr_windowed_bb)))/5
                    img = axs[i].imshow(dimer_distr_windowed_bb[step], extent=[r_bins[0], r_bins[-1], r_bins[0], r_bins[-1]], vmin = 0, vmax = v_max, cmap = 'gnuplot')
                    axs[i].set(title = r'$D_{bb}$' + f"  at stage {i+1}")
                if (len(params['red_particle_idx']) > 0):
                    v_max = np.max(np.concatenate((dimer_distr_windowed_rr)))/5
                    img = axs[i].imshow(dimer_distr_windowed_rr[step], extent=[r_bins[0], r_bins[-1], r_bins[0], r_bins[-1]], vmin = 0, vmax = v_max, cmap = 'gnuplot')
                    axs[i].set(title = r'$D_{rr}$' + f"  at stage {i+1}")
            for i, ax in enumerate(axs.flatten()):
                ax.text(0.0, 1.0, f"{params['letter_labels'][i]}", transform=(ax.transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)), fontsize='medium', va='bottom')
            axs[0].set(xticks = [- 45, - 20, 0, 20, 45], yticks = [- 45, -20, 0, 20, 45], ylabel = 'y [mm]')

            fig.colorbar(img, ax=axs, orientation='vertical',  format='%.0e', fraction=0.046, pad=0.04)
            if save_plots:
                plt.savefig(f"{params['res_path']}/dimer_analysis/dimer_distributions_{params['n_stages']}.png", bbox_inches='tight')
                plt.savefig(f"{params['pdf_res_path']}/dimer_analysis/dimer_distributions_{params['n_stages']}.pdf", bbox_inches='tight')
            if show_plots:
                plt.show()
            else:
                plt.close()
        
            if animated_plot_results:
                fig, ax = plt.subplots(1, 1, figsize = (15, 15))
                anim_running = True

                def update(step):
                    if len(params['blue_particle_idx']) > 0:
                        img.set_data(dimer_distr_windowed_bb[step])
                        title.set_text(r'$D_{bb}$' + f"at {int((params['startFrames'][step] + params['window_length']*params['fps'])/params['fps'])} s")
                    if len(params['red_particle_idx']) > 0:
                        img.set_data(dimer_distr_windowed_rr[step])
                        title.set_text(r'$D_{rr}$' + f"at {int((params['startFrames'][step] + params['window_length']*params['fps'])/params['fps'])} s")
                    return img, title
                        
                if len(params['blue_particle_idx']) > 0:
                    img = ax.imshow(dimer_distr_windowed_bb[0], extent=[r_bins[0], r_bins[-1], r_bins[0], r_bins[-1]], vmin = 0, vmax = v_max, cmap = 'gnuplot')
                    title = ax.set_title(r'$D_{bb}$' + f"at {int((params['startFrames'][0] + params['window_length']*params['fps'])/params['fps'])} s")

                if len(params['red_particle_idx']) > 0:
                    img = ax.imshow(dimer_distr_windowed_rr[0], extent=[r_bins[0], r_bins[-1], r_bins[0], r_bins[-1]], vmin = 0, vmax = v_max, cmap = 'gnuplot')
                    title = ax.set_title(r'$D_{rr}$' + f"at {int((params['startFrames'][0] + params['window_length']*params['fps'])/params['fps'])} s")

                fig.canvas.mpl_connect('button_press_event', onClick)
                ani = FuncAnimation(fig, update, np.arange(0, params['n_windows'], 10), interval = 5, blit=False)
                writer = FFMpegWriter(fps = 10, metadata = dict(artist='skandiz'), extra_args=['-vcodec', 'libx264'])
                if save_plots:
                    ani.save(f"./{params['res_path']}/dimer_analysis/dimer_distributions_animation.mp4", writer = writer, dpi = 300)
                if show_plots:
                    plt.show()
                else:
                    plt.close()
                
        
        if (len(params['blue_particle_idx']) > 0) & (len(params['red_particle_idx']) > 0) :
            v_max = np.max(np.concatenate((dimer_distr_windowed_bb, dimer_distr_windowed_rr, dimer_distr_windowed_br, dimer_distr_windowed_rb)))/5
                
            fig, axs = plt.subplots(4, params['n_stages'], figsize = (18, 14), sharex=True, sharey=True)
            for i, step in enumerate(params['steps_plot']):
                img = axs[0, i].imshow(dimer_distr_windowed_bb[step], extent=[r_bins[0], r_bins[-1], r_bins[0], r_bins[-1]], vmin = 0, vmax = v_max, cmap = 'gnuplot')
                axs[0, i].set(title = r'$D_{bb}$' + f"  at stage {i+1}")
                axs[1, i].imshow(dimer_distr_windowed_rr[step], extent=[r_bins[0], r_bins[-1], r_bins[0], r_bins[-1]], vmin = 0, vmax = v_max, cmap = 'gnuplot')
                axs[1, i].set(title = r'$D_{rr}$' + f"  at stage {i+1}")
                axs[2, i].imshow(dimer_distr_windowed_br[step], extent=[r_bins[0], r_bins[-1], r_bins[0], r_bins[-1]], vmin = 0, vmax = v_max, cmap = 'gnuplot')
                axs[2, i].set(title = r'$D_{br}$' + f"  at stage {i+1}")
                axs[3, i].imshow(dimer_distr_windowed_rb[step], extent=[r_bins[0], r_bins[-1], r_bins[0], r_bins[-1]], vmin = 0, vmax = v_max, cmap = 'gnuplot')
                axs[3, i].set(title = r'$D_{rb}$' + f"  at stage {i+1}")
                axs[3, i].set(xlabel = 'x [mm]')

            for i, ax in enumerate(axs.flatten()):
                ax.text(0.0, 1.0, f"{params['letter_labels'][i]}", transform=(ax.transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)), fontsize='medium', va='bottom')
            axs[0, 0].set(xticks = [- 45, - 20, 0, 20, 45], yticks = [- 45, -20, 0, 20, 45])

            axs[0, 0].set(ylabel = 'y [mm]')
            axs[1, 0].set(ylabel = 'y [mm]')
            axs[2, 0].set(ylabel = 'y [mm]')
            axs[3, 0].set(ylabel = 'y [mm]')
            fig.colorbar(img, ax=axs, orientation='vertical',  format='%.0e', fraction=0.046, pad=0.04)
            if save_plots:
                plt.savefig(f"{params['res_path']}/dimer_analysis/dimer_distributions_{params['n_stages']}.png", bbox_inches='tight')
                plt.savefig(f"{params['pdf_res_path']}/dimer_analysis/dimer_distributions_{params['n_stages']}.pdf", bbox_inches='tight')
            if show_plots:
                plt.show()
            else:
                plt.close()
                
                
            fig, axs = plt.subplots(4, params['n_stages'], figsize = (18, 14), sharex=True, sharey=True)
            for i, step in enumerate(params['steps_plot']):
                img = axs[0, i].imshow(dimer_distr_windowed_bb[step], extent=[r_bins[0], r_bins[-1], r_bins[0], r_bins[-1]], vmin = 0, vmax = v_max, cmap = 'gnuplot')
                axs[0, i].set(title =  f"Stage {i+1}")
                axs[1, i].imshow(dimer_distr_windowed_rr[step], extent=[r_bins[0], r_bins[-1], r_bins[0], r_bins[-1]], vmin = 0, vmax = v_max, cmap = 'gnuplot')
                axs[2, i].imshow(dimer_distr_windowed_br[step], extent=[r_bins[0], r_bins[-1], r_bins[0], r_bins[-1]], vmin = 0, vmax = v_max, cmap = 'gnuplot')
                axs[3, i].imshow(dimer_distr_windowed_rb[step], extent=[r_bins[0], r_bins[-1], r_bins[0], r_bins[-1]], vmin = 0, vmax = v_max, cmap = 'gnuplot')
                
            for i, ax in enumerate(axs.flatten()):
                ax.text(0.0, 1.0, f"{params['letter_labels'][i]}", transform=(ax.transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)), fontsize='medium', va='bottom')
                
            axs[0, 0].set(ylabel = r'$D_{bb}$')
            axs[1, 0].set(ylabel = r'$D_{rr}$')
            axs[2, 0].set(ylabel = r'$D_{br}$')
            axs[3, 0].set(ylabel = r'$D_{rb}$')
            axs[0, 0].set(xticks = [], yticks = [])
            plt.suptitle(f"Dimer distribution of system {params['system_name']}")

            plt.tight_layout()
            if save_plots:
                plt.savefig(f"{params['res_path']}/dimer_analysis/dimer_distributions_{params['n_stages']}_v2.png", bbox_inches='tight')
                plt.savefig(f"{params['pdf_res_path']}/dimer_analysis/dimer_distributions_{params['n_stages']}_v2.pdf", bbox_inches='tight')
            if show_plots:
                plt.show()
            else:
                plt.close()
                    
            if animated_plot_results:
                
                test_frame = params['frames_stages'][1]

                blue_pos = trajectories.loc[trajectories.particle.isin(params['blue_particle_idx']), ['x','y']].values.reshape(params['n_frames'], len(params['blue_particle_idx']), 2)
                red_pos = trajectories.loc[trajectories.particle.isin(params['red_particle_idx']),  ['x','y']].values.reshape(params['n_frames'], len(params['red_particle_idx']), 2)

                coords_blue = blue_pos[test_frame]
                coords_red = red_pos[test_frame]
                coords = np.concatenate([coords_blue, coords_red], axis = 0)
                kd_blue = KDTree(coords_blue)

                d_red_blue, ids_red_blue = kd_blue.query(coords_red, k = 1)
                angles_red_blue = -np.arctan2(coords_blue[ids_red_blue, 1] - coords_red[:, 1], coords_blue[ids_red_blue, 0] - coords_red[:, 0])

                fig, ax = plt.subplots(1, 1, figsize = (10, 10), sharex=True, sharey=True)
                anim_running = True

                def update(x):
                    for test_droplet in range(25):            
                        temp = coords - x * coords_red[test_droplet, :]
                        x_new = temp[:, 0] * np.cos(x * angles_red_blue[test_droplet]) - temp[:, 1] * np.sin(x * angles_red_blue[test_droplet])
                        y_new = temp[:, 0] * np.sin(x * angles_red_blue[test_droplet]) + temp[:, 1] * np.cos(x * angles_red_blue[test_droplet])
                        temp = np.stack([x_new, y_new], axis = 1)
                        scat3[test_droplet].set_offsets(temp)
                        
                scat, scat1, scat2, scat3 = {}, {}, {}, {}
                for test_droplet in range(25):
                    scat3[test_droplet] = ax.scatter(coords[:, 0], coords[:, 1], color = params['colors'], alpha = 0.3)

                title = ax.set_title('red-blue')

                ax.grid(linewidth = 0.5)
                ax.set(xlim = (-params['resolution'], params['resolution']), ylim = (-params['resolution'], params['resolution']))

                fig.canvas.mpl_connect('button_press_event', onClick)
                ani = FuncAnimation(fig, update, np.linspace(0, 1, 100), interval = 10, blit=False)
                writer = FFMpegWriter(fps = 10, metadata = dict(artist='skandiz'), extra_args=['-vcodec', 'libx264'])
                ani.save(f"./{params['res_path']}/dimer_analysis/rototranslations_{test_frame}_v2.mp4", writer = writer, dpi = 300)
                plt.close()
                
                fig, axs = plt.subplots(2, 2, figsize = (15, 15))
                anim_running = True

                def update(step):
                    img.set_data(dimer_distr_windowed_bb[step])
                    img1.set_data(dimer_distr_windowed_rr[step])
                    img2.set_data(dimer_distr_windowed_br[step])
                    img3.set_data(dimer_distr_windowed_rb[step])
                    title.set_text(r'$D_{bb}$' + f"at {int((params['startFrames'][step] + params['window_length']*params['fps'])/params['fps'])} s")
                    title1.set_text(r'$D_{rr}$' + f"at {int((params['startFrames'][step] + params['window_length']*params['fps'])/params['fps'])} s")
                    title2.set_text(r'$D_{br}$' + f"at {int((params['startFrames'][step] + params['window_length']*params['fps'])/params['fps'])} s")
                    title3.set_text(r'$D_{rb}$' + f"at {int((params['startFrames'][step] + params['window_length']*params['fps'])/params['fps'])} s")
                    
                img = axs[0, 0].imshow(dimer_distr_windowed_bb[0], extent=[r_bins[0], r_bins[-1], r_bins[0], r_bins[-1]], vmin = 0, vmax = v_max, cmap = 'gnuplot')
                img1 = axs[0, 1].imshow(dimer_distr_windowed_rr[0], extent=[r_bins[0], r_bins[-1], r_bins[0], r_bins[-1]], vmin = 0, vmax = v_max, cmap = 'gnuplot')
                img2 = axs[1, 0].imshow(dimer_distr_windowed_br[0], extent=[r_bins[0], r_bins[-1], r_bins[0], r_bins[-1]], vmin = 0, vmax = v_max, cmap = 'gnuplot')
                img3 = axs[1, 1].imshow(dimer_distr_windowed_rb[0], extent=[r_bins[0], r_bins[-1], r_bins[0], r_bins[-1]], vmin = 0, vmax = v_max, cmap = 'gnuplot')
                title = axs[0, 0].set_title(r'$D_{bb}$' + f"at {int((params['startFrames'][0] + params['window_length']*params['fps'])/params['fps'])} s")
                title1 = axs[0, 1].set_title(r'$D_{rr}$' + f"at {int((params['startFrames'][0] + params['window_length']*params['fps'])/params['fps'])} s")
                title2 = axs[1, 0].set_title(r'$D_{br}$' + f"at {int((params['startFrames'][0] + params['window_length']*params['fps'])/params['fps'])} s")
                title3 = axs[1, 1].set_title(r'$D_{rb}$' + f"at {int((params['startFrames'][0] + params['window_length']*params['fps'])/params['fps'])} s")
                plt.tight_layout()
                fig.canvas.mpl_connect('button_press_event', onClick)
                ani = FuncAnimation(fig, update, np.arange(0, params['n_windows'], 10), interval = 5, blit=False)
                writer = FFMpegWriter(fps = 10, metadata = dict(artist='skandiz'), extra_args=['-vcodec', 'libx264'])
                if save_plots:
                    ani.save(f"./{params['res_path']}/dimer_analysis/dimer_distributions_animation.mp4", writer = writer, dpi = 300)
                if show_plots:
                    plt.show()
                else:
                    plt.close()

    if (len(params['blue_particle_idx']) > 0) & (len(params['red_particle_idx']) > 0):
        return (dimer_distr_windowed_bb, dimer_distr_windowed_rr, dimer_distr_windowed_br, dimer_distr_windowed_rb), r_bins, v_max
    elif (len(params['blue_particle_idx']) > 0) & (len(params['red_particle_idx']) == 0):
        return (dimer_distr_windowed_bb, None, None, None), r_bins, v_max
    elif (len(params['blue_particle_idx']) == 0) & (len(params['red_particle_idx']) > 0):
        return (None, dimer_distr_windowed_rr, None, None), r_bins, v_max