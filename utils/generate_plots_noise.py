from matplotlib.collections import LineCollection
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
from matplotlib.colors import Normalize, ListedColormap
from matplotlib import container
import sys, os

ROOT_FOLDER = './sim_results'
EXPORT_FOLDER =  './exported_plots'

def generate_viewing_error_plot(all_drone_data, ax):
    WIDTH_X, WIDHT_Y = 0.1, 0.2
    ax.set_title('Drones viewing direction error', loc='left')
    # Labelling
    ax.set_xlabel('# neighbors')
    ax.set_ylabel('Drone index')
    ax.set_zlabel('Error (°)', labelpad=-30)
    cmaps = ['Purples', 'Blues', 'Greens', 'Oranges', 'Greys']
    # Create the bins
    for n, key in enumerate(all_drone_data.keys()):
        nb_drones = len(all_drone_data[key][0][0])
        x_range = np.array([i+(-2+n)*(WIDTH_X) for i in range(from_, to+1)])
        y_range = np.arange(nb_drones) - WIDHT_Y/2
        # Find the average error for each drone
        data = [np.array(all_drone_data[key][i]) for i in range(len(all_drone_data[key]))]
        errors = []
        dist_to_hull = []
        nb_tests = len(data)
        total_bars = nb_tests * nb_drones
        xpos, ypos, zpos, dx, dy, dz = np.zeros((6, total_bars))
        idx_ptr = 0
        for i in range(nb_tests):
            err = np.mean(data[i][:, :, 3], axis=0)
            dist = np.mean(np.abs(data[i][:, :, 2]), axis=0)
            dist_sorted_indices = np.argsort(dist, axis=0)
            errors.append(err[dist_sorted_indices])
            dist_to_hull.append(dist[dist_sorted_indices])

            # Create the bins
            # loop over all drone indices    
            nb_idx = len(y_range)
            xpos[idx_ptr:idx_ptr+nb_idx] = x_range[i]*np.ones(nb_idx)
            ypos[idx_ptr:idx_ptr+nb_idx] = y_range

            # Construct arrays with the dimensions for the bars.
            dx[idx_ptr:idx_ptr+nb_idx] = np.ones(nb_idx) * WIDTH_X
            dy[idx_ptr:idx_ptr+nb_idx] = np.ones(nb_idx) * WIDHT_Y
            dz[idx_ptr:idx_ptr+nb_idx] = errors[-1]
            idx_ptr += nb_idx

        # Create color map with dist to center (on average)
        colormap = mpl.colormaps[cmaps[n]]
        new_cm = ListedColormap(colormap(np.linspace(0.25, 0.75, 256)))
        #max_height = np.max(dist_to_hull)
        #min_height = np.min(dist_to_hull)
        dist_to_hull = np.concatenate(dist_to_hull)
        #colors = new_cm((dist_to_hull - min_height) / (max_height-min_height))
        colors = new_cm(dist_to_hull)
        ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors, zsort='average', label=f'{key}', shade=False, edgecolor='black')
        # Add average over all drones as 2d line
        avg_errors = np.mean(errors, axis=1)
        ax.plot(x_range, avg_errors, zs=nb_drones, zdir='y', label=f'Avg error {key}', linewidth=2, color=new_cm(175))
        ax.legend(fontsize='x-small', loc='best')
        #ax.set_box_aspect(aspect=None, zoom=0.95)
        if n == 0:
            cbar = plt.colorbar(cm.ScalarMappable(Normalize(vmin=0, vmax=1), cmap=new_cm), ax=ax, orientation='vertical', fraction=0.02, pad=0.0, aspect=30, shrink=0.4)
            cbar.set_label('Normalized distance to convex hull center', fontsize='small')
        elif n == len(all_drone_data.keys())-1:
            plt.colorbar(cm.ScalarMappable(Normalize(vmin=0, vmax=1), cmap=new_cm), ax=ax, orientation='vertical', fraction=0.02, pad=0.04, aspect=30, shrink=0.4, ticks=[], boundaries=None, drawedges=False)
        else:
            plt.colorbar(cm.ScalarMappable(Normalize(vmin=0, vmax=1), cmap=new_cm), ax=ax, orientation='vertical', fraction=0.02, pad=0.0, aspect=30, shrink=0.4, ticks=[], boundaries=None, drawedges=False)

    #cbar = plt.colorbar(cm.ScalarMappable(Normalize(vmin=0, vmax=1), cmap=new_cm), ax=ax, orientation='vertical', fraction=0.02, pad=0.04, aspect=30, shrink=0.4, ticks=[], boundaries=None, drawedges=False)
    #cbar.set_label('Normalized distance to convex hull center', fontsize='small')

def generate_avg_viewing_error_plot(all_drone_data, ax):
    ax.set_title('Average viewing direction error')
    ax.set_xlabel('# neighbors')
    ax.set_ylabel('Error (°)')
    colors = ['purple', 'blue', 'green', 'darkorange', 'black']
    for n, key in enumerate(all_drone_data.keys()):
        data = [np.array(all_drone_data[key][i]) for i in range(len(all_drone_data[key]))]
        errors = []
        # Find index of center drone in average
        for i in range(NB_TESTS):
            err = np.mean(data[i][:, :, 3], axis=0)
            dist = np.mean(np.abs(data[i][:, :, 2]), axis=0)
            dist_sorted_indices = np.argsort(dist, axis=0)
            errors.append(err[dist_sorted_indices])
        avg_errors = np.mean(np.array(errors)[:,1:], axis=1)
        std_errors = np.std(np.array(errors)[:,1:], axis=1)
        ax.errorbar(np.arange(from_, to+1, steps), avg_errors, yerr=[np.clip(std_errors, 0, avg_errors), std_errors], fmt='--', label=f"{key.replace('_', ' ')}", markersize=5, marker='o', ecolor=colors[n], color=colors[n], capsize=5, capthick=2)
    handles, labels = ax.get_legend_handles_labels()
    handles = [h[0] if isinstance(h, container.ErrorbarContainer) else h for h in handles]
    ax.legend(handles, labels, fontsize=7, draggable=True, markerscale=0.5, loc='best')

def generate_coverage_plot(all_swarm_data, ax):
    ax.set_title('Mean and std of the coverage')
    ax.set_xlabel('# neighbors')
    ax.set_ylabel('Coverage %')
    colors = ['purple', 'blue', 'green', 'darkorange', 'black']
    for n, key in enumerate(all_swarm_data.keys()):
        # Find the average coverage for each drone
        coverage_mean = np.array([np.mean(all_swarm_data[key][i]['viewing_coverage'], axis=0) for i in range(NB_TESTS)])
        coverage_std = np.array([np.std(all_swarm_data[key][i]['viewing_coverage'], axis=0) for i in range(NB_TESTS)])
        ax.errorbar(np.arange(from_, to+1, steps), coverage_mean*100, yerr=[coverage_std*100, np.clip(coverage_std, 0, 1-coverage_mean)*100], fmt='--', label=f"{key.replace('_', ' ')}", markersize=5, marker='o', ecolor=colors[n], color=colors[n], capsize=5, capthick=2)
    # Filter out error bars from legend
    handles, labels = ax.get_legend_handles_labels()
    handles = [h[0] if isinstance(h, container.ErrorbarContainer) else h for h in handles]
    ax.legend(handles, labels, fontsize=7, draggable=True, markerscale=0.5, loc='best')

def generate_overlap_plot(all_swarm_data, ax):
    ax.set_title('Mean and std of the drones overlap')
    ax.set_xlabel('# neighbors')
    ax.set_ylabel('Overlap (°)')
    colors = ['purple', 'blue', 'green', 'darkorange', 'black']
    for n, key in enumerate(all_swarm_data.keys()):
        # Find the average coverage for each drone
        overlap_mean = np.array([np.mean(all_swarm_data[key][i]['viewing_overlap']*180/np.pi, axis=0) for i in range(NB_TESTS)])
        overlap_std = np.array([np.std(all_swarm_data[key][i]['viewing_overlap']*180/np.pi, axis=0) for i in range(NB_TESTS)])
        ax.errorbar(np.arange(from_, to+1, steps), overlap_mean, yerr=[np.clip(overlap_std, 0, overlap_mean), overlap_std], fmt='--', label=f"{key.replace('_', ' ')}", markersize=5, marker='o', ecolor=colors[n], color=colors[n], capsize=5, capthick=2)
    # Filter out error bars from legend
    handles, labels = ax.get_legend_handles_labels()
    handles = [h[0] if isinstance(h, container.ErrorbarContainer) else h for h in handles]
    ax.legend(handles, labels, fontsize=7, draggable=True, markerscale=0.5, loc='best')

def generate_timing_viewing_plot(timing_data, ax, x_range):
    ax.set_title('Viewing computation time')
    ax.set_xlabel('# neighbors')
    ax.set_ylabel('Average viewing computation time (ms))')
    colors = ['purple', 'blue', 'green', 'darkorange', 'black']
    for n, key in enumerate(all_drone_data.keys()):
        data = timing_data[key]
        # Compute average timings with std
        avg_timing = np.array([np.mean(np.array(data[i])[:, 1], axis=0) for i in range(NB_TESTS)])
        std_timing = np.array([np.std(np.array(data[i])[:, 1], axis=0) for i in range(NB_TESTS)])
        #ax.errorbar(x_range, avg_timing*1000, yerr=[np.clip(std_timing, 0, avg_timing)*1000, std_timing*1000], fmt='--', label=f"{key.replace('_', ' ')}", markersize=5, marker='o', ecolor=colors[n], color=colors[n], capsize=5, capthick=2)
        ax.plot(x_range, avg_timing*1000, label=f"{key.replace('_', ' ')}", markersize=5, marker='o', color=colors[n], linestyle='--')  
    # Filter out error bars from legend
    handles, labels = ax.get_legend_handles_labels()
    handles = [h[0] if isinstance(h, container.ErrorbarContainer) else h for h in handles]
    ax.legend(handles, labels, fontsize=7, draggable=True, markerscale=0.5, loc='best')

def generate_timings_plot(timings, ax, x_range):
    ax.set_title('Timings')
    ax.set_xlabel('# neighbors')
    ax.set_ylabel('Computation time (ms)')
    # Compute average timings
    timings_neighborhood = np.mean(timings[:, :, 0], axis=1)*1000
    timings_viewing = np.mean(timings[:, :, 1], axis=1)*1000
    timings_coverage = np.mean(timings[:, :, 2], axis=1)*1000
    timings_total = timings_neighborhood + timings_viewing + timings_coverage
    ax.bar(x_range, timings_neighborhood, width=0.6, label='Neighborhood')
    ax.bar(x_range, timings_viewing, width=0.6, label='Viewing', bottom=timings_neighborhood)
    ax.bar(x_range, timings_coverage, width=0.6, label='Coverage', bottom=timings_neighborhood+timings_viewing)
    for i in range(to-from_+1):
        value_1 = round(timings_neighborhood[i]/timings_total[i] ,2)
        ax.text(x_range[i], timings_neighborhood[i]/2, value_1, ha = 'center')
        value_2 = round(timings_viewing[i]/timings_total[i] ,2)
        ax.text(x_range[i], timings_neighborhood[i]+timings_viewing[i]/2, value_2, ha = 'center')
        value_3 = round(1 - value_1 - value_2, 2)
        ax.text(x_range[i], timings_neighborhood[i]+timings_viewing[i]+timings_coverage[i]/2, value_3, ha = 'center')
    ax.legend()

def generate_swarm_center_plot(swarm_data, ax):
    ax.set_title('Swarm center position')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    for n, key in enumerate(all_drone_data.keys()):
        data = [np.array(swarm_data[key][i]['centers']) for i in range(len(swarm_data[key]))]
        # Compute center x,y
        for i in range(len(data)):
            center_x = data[i][:, 0]
            center_y = data[i][:, 1]
            ax.plot(center_x, center_y, label=f'Run {n*len(data)+i+1}')
        # ax.plot(center_x, center_y, 'b-', linewidth=2)
    ax.grid(True)
    #ax.legend()

def generate_trajectory_corverage(swarm_data, axs, fig):
    # Loop for all tests with neighbors
    cov_min = np.min([np.min(swarm_data[i]['viewing_coverage']) for i in range(NB_TESTS)])
    cov_max = np.max([np.max(swarm_data[i]['viewing_coverage']) for i in range(NB_TESTS)])
    for i in range(NB_TESTS):
        coverage = swarm_data[i]['viewing_coverage']*100
        center = swarm_data[i]['centers'][:,:2]
        axs[i].set_title(f'# Neighbors: {i+from_}')
        # Create a mapping for colors
        segments = np.array([[center[i], center[i+1]] for i in range(len(center)-1)])
        norm = plt.Normalize(cov_min*100, cov_max*100)
        lc = LineCollection(segments, cmap='viridis', norm=norm)
        lc.set_array(coverage)
        lc.set_linewidth(2)
        axs[i].add_collection(lc)
        # axs[i].set_xlim(-5,5)
        # axs[i].set_ylim(-5,5)
        axs[i].set_xlim(center[:, 0].min()*1.1, center[:, 0].max()*1.1)
        axs[i].set_ylim(center[:, 1].min()*1.1, center[:, 1].max()*1.1)
        fig.colorbar(lc, ax=axs[i], label='Coverage %')

if __name__ == '__main__':
    # Read arguments
    nb_args = len(sys.argv)
    if nb_args < 2:
        print('Please provide at least one argument --> --visual or --export')
        sys.exit(1)
    
    cmd = sys.argv[1]
    folder = ROOT_FOLDER
    export_mode = False
    if nb_args == 3:
        folder = os.path.join(ROOT_FOLDER, sys.argv[2])
    if cmd == '--visual':
        print(f'Visualizing the results of folder "{folder}"')
    elif cmd == '--export':
        export_mode = True
        os.makedirs(EXPORT_FOLDER, exist_ok=True)
        print(f'Exporting the results of folder "{folder}" to "{EXPORT_FOLDER}"')
    else:
        print('Please provide a valid argument --> --visual or --export')
        sys.exit(1)
    
    # Loop through subfolders if any
    if nb_args < 3:
        subfolders = os.listdir(folder)
    else:
        subfolders = [sys.argv[i] for i in range(2, nb_args)]
    
    for i in range(len(subfolders)):
        subfolder = subfolders[i]
        print(f"Generating plots... {i+1}/{len(subfolders)}", end="\r")
        sim_files = os.listdir(os.path.join(ROOT_FOLDER, subfolder))
        sim_files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
        # Read data dict (either if visualizing or exporting)
        filepaths = [os.path.join(ROOT_FOLDER, subfolder, f) for f in sim_files]
        try:
            from_ = int(sim_files[0].strip('.npy')[-1])
            to = int(sim_files[-1].strip('.npy')[-1])
            steps = int(sim_files[4].strip('.npy')[-1]) - from_
        except ValueError:
            print('Could not determine range of tests: see file names convention')
            sys.exit(1)
        NB_TESTS = (to - from_)//steps + 1
        all_drone_data = dict()
        timing_data = dict()
        all_swarm_data = dict()
        params = {}
        try:
            for p, f in zip(filepaths, sim_files):
                data = np.load(p, allow_pickle=True).item()
                w = f.split('_')
                if "average" in w:
                    wkey = "average"
                elif "tangent" in w:
                    wkey = "tangent_plane"
                elif "adjacent" in w:
                    wkey = "convex_hull_adjacent"
                elif "visible" in w:
                    wkey = "convex_hull_visible"
                elif "outter" in w:
                    idx = w.index("outter")
                    wkey = f'outter_{w[idx+1]}'
                if wkey not in all_drone_data:
                    all_drone_data[wkey] = []
                    timing_data[wkey] = []
                    all_swarm_data[wkey] = []
                all_drone_data[wkey].append(data['drone_data'])
                timing_data[wkey].append(data['timings'])
                all_swarm_data[wkey].append(data['swarm_data'])
                params.update({wkey: data['params']})
        except FileNotFoundError:
            print('File not found')

    if export_mode:
        # Create export subfolder if needed
        os.makedirs(os.path.join(EXPORT_FOLDER, subfolder), exist_ok=True)
        # Export 3D plot for viewing error
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        generate_viewing_error_plot(all_drone_data, ax)
        fig.tight_layout()
        fig.savefig(os.path.join(EXPORT_FOLDER, subfolder, f"viewing_error_3D.png"))

        # Export each plot individually
        x_range = np.arange(from_, to+1)
        elems_title = subfolder.split('_')
        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot(111)
        generate_coverage_plot(all_swarm_data, ax)
        fig.tight_layout()
        fig.savefig(os.path.join(EXPORT_FOLDER, subfolder, f"coverage.png"))

        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot(111)
        generate_overlap_plot(all_swarm_data, ax)
        fig.tight_layout()
        fig.savefig(os.path.join(EXPORT_FOLDER, subfolder, f"overlap.png"))

        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot(111)
        generate_swarm_center_plot(all_swarm_data, ax)
        fig.tight_layout()
        fig.savefig(os.path.join(EXPORT_FOLDER, subfolder, f"swarm_center.png"))

        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot(111)
        generate_timing_viewing_plot(timing_data, ax, x_range)
        fig.tight_layout()
        fig.savefig(os.path.join(EXPORT_FOLDER, subfolder, f"viewing_timings.png"))

        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot(111)
        generate_avg_viewing_error_plot(all_drone_data, ax)
        fig.tight_layout()
        fig.savefig(os.path.join(EXPORT_FOLDER, subfolder, f"avg_viewing_error.png"))

        # Swarm trajectory
        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot(111)
        generate_swarm_center_plot(all_swarm_data, ax)
        fig.tight_layout()
        fig.savefig(os.path.join(EXPORT_FOLDER, subfolder, f"swarm_trajectory.png"))
        

        for k in all_drone_data.keys():
            fig = plt.figure(figsize=(12, 8))
            fig.suptitle(f"Coverage during trajectory with viewing metric {' '.join(k.split('_'))}", fontsize=16)
            gs = fig.add_gridspec(3,2)
            axs = [fig.add_subplot(gs[i, j]) for j in range(2) for i in range(3)]
            generate_trajectory_corverage(all_swarm_data[k], axs, fig)
            fig.tight_layout()
            fig.savefig(os.path.join(EXPORT_FOLDER, subfolder, f"coverage_trajectory_{k}.png"))

    else:
        # Create the plots
        fig = plt.figure(figsize=(12, 8))
        elems_title = subfolder.split('_')
        fig.suptitle(f"Results for testing {' '.join(elems_title[:-1])} metric", fontsize=16)
        gs = fig.add_gridspec(3,5)
        ax1 = fig.add_subplot(gs[0:3,0:3], projection='3d')
        ax2 = fig.add_subplot(gs[0,3:5])
        ax3 = fig.add_subplot(gs[1,3:5])
        ax4 = fig.add_subplot(gs[2,3:5])
        x_range = np.arange(from_, to+1)

        generate_viewing_error_plot(all_drone_data, ax1)
        generate_coverage_plot(all_swarm_data, ax2)
        # generate_overlap_plot(all_swarm_data, ax3)
        generate_timing_viewing_plot(timing_data, ax3, x_range)
        # generate_swarm_center_plot(all_swarm_data, ax4)
        generate_avg_viewing_error_plot(all_drone_data, ax4)

        # for k in all_drone_data.keys():
        #     fig2 = plt.figure(figsize=(12, 8))
        #     fig2.suptitle(f"Coverage during trajectory with viewing metric {' '.join(k.split('_'))}", fontsize=16)
        #     gs2 = fig2.add_gridspec(3,2)
        #     axs = [fig2.add_subplot(gs2[i, j]) for j in range(2) for i in range(3)]

        #     generate_trajectory_corverage(all_swarm_data[k], axs, fig2)
        #     fig2.tight_layout()
        # if not export_mode and nb_args == 3:
        #     # Create metrics summary
        #     swarm_spread = data[0]['params']['swarm_spread']
        #     r_coh = data[0]['params']['r_coh']
        #     neighbors_metric = data[0]['params']['neighbors']['metric']
        #     neighbors_count = data[0]['params']['neighbors']['count']
        #     neighbors_range = data[0]['params']['neighbors']['sensing_range']
        #     neighbors_r_agent = data[0]['params']['neighbors']['r_agent']
        #     noise = data[0]['params']['noise']
        #     viewing_dim = '2D' if data[0]['params']['viewing_metric']['dim'] == 2 else '3D'
        #     fig2, ax = plt.subplots(1, 1)
        #     ax.set_title('Metrics summary', fontweight='bold', fontstyle='oblique', fontsize=16)
        #     ax.axis('off')
        #     ax.text(0, 0.9, f'Number of drones: {nb_drones}', fontsize=12)
        #     ax.text(0, 0.8, f'Swarm spread: {swarm_spread:.2f}', fontsize=12)
        #     ax.text(0, 0.7, f'Cohesion radius: {r_coh:.2f}', fontsize=12)
        #     ax.text(0, 0.6, f'Neighbors:  ', fontsize=12)
        #     ax.text(0.25, 0.6, f'Metric: {neighbors_metric}', fontsize=12)
        #     ax.text(0.25, 0.53, f'Count: {neighbors_count}', fontsize=12)
        #     ax.text(0.25, 0.46, f'Sensing range: {neighbors_range:.2f}', fontsize=12)
        #     ax.text(0.25, 0.39, f'Agent radius: {neighbors_r_agent:.3f}', fontsize=12)
        #     ax.text(0, 0.3, f'Noise: ', fontsize=12)
        #     ax.text(0.25, 0.3, f'type: {noise["type"]}', fontsize=12)
        #     ax.text(0.25, 0.23, f'dist: {noise["param_dist"]:.2f}', fontsize=12)
        #     ax.text(0.25, 0.16, f'cone: {noise["param_dir"]:.2f}', fontsize=12)
        #     ax.text(0, 0.05, f'Viewing metric dim: {viewing_dim}', fontsize=12)
        #     fig2.tight_layout()
        
        fig.tight_layout()

    print('\nDONE GENERATING PLOTS!')
    if export_mode:
        print(f'Plots have been exported to "{EXPORT_FOLDER}/{subfolder}"!')
    else:
        plt.show()
