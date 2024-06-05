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
    ax.set_title('Drones viewing direction error', loc='left')
    # Labelling
    ax.set_xlabel('Swarm count')
    ax.set_ylabel('Error (°)')
    colors = ['purple', 'blue', 'green', 'darkorange', 'black']
    x_range = np.array([5,10,15,20,25,30])
    # Create the bins
    for n, key in enumerate(all_drone_data.keys()):
        # Find the average error for each drone
        data = [np.array(all_drone_data[key][i]) for i in range(len(all_drone_data[key]))]
        avg_errors = [np.mean(data[i][:,:,3], axis=(0,1)) for i in range(NB_TESTS)]
        std_errors = [np.std(data[i][:,:,3], axis=(0,1)) for i in range(NB_TESTS)]
        ax.errorbar(x_range, avg_errors, yerr=std_errors, fmt='--', label=f"{key.replace('_', ' ')}", markersize=5, marker='o', ecolor=colors[n], color=colors[n], capsize=5, capthick=2)
        # Filter out error bars from legend
    handles, labels = ax.get_legend_handles_labels()
    handles = [h[0] if isinstance(h, container.ErrorbarContainer) else h for h in handles]
    ax.legend(handles, labels, fontsize=7, draggable=True, markerscale=0.5, loc='best')

def generate_avg_num_neighbors_plot(all_drone_data, ax):
    x_range = np.array([5,10,15,20,25,30])
    ax.set_title('Average number of neighbors')
    ax.set_xlabel('Swarm count')
    ax.set_ylabel('Neighbors')
    colors = ['purple', 'blue', 'green', 'darkorange', 'black']
    for n, key in enumerate(all_drone_data.keys()):
        data = [np.array(all_drone_data[key][i]) for i in range(len(all_drone_data[key]))]
        neighbors = []
        for i in range(NB_TESTS):
            neighbors.append(np.mean(data[i][:, :, 0], axis=(0,1)))
        ax.plot(x_range, neighbors, label=f"{key.replace('_', ' ')}", color=colors[n], linestyle='--', marker='o', markersize=5)
    ax.legend(fontsize='small')

def generate_coverage_plot(all_swarm_data, ax):
    x_range = np.array([5,10,15,20,25,30])
    ax.set_title('Mean and std of the coverage')
    ax.set_xlabel('Swarm count')
    ax.set_ylabel('Coverage %')
    colors = ['purple', 'blue', 'green', 'darkorange', 'black']
    for n, key in enumerate(all_swarm_data.keys()):
        # Find the average coverage for each drone
        coverage_mean = np.array([np.mean(all_swarm_data[key][i]['viewing_coverage'], axis=0) for i in range(NB_TESTS)])
        coverage_std = np.array([np.std(all_swarm_data[key][i]['viewing_coverage'], axis=0) for i in range(NB_TESTS)])
        ax.errorbar(x_range, coverage_mean*100, yerr=[coverage_std*100, np.clip(coverage_std, 0, 1-coverage_mean)*100], fmt='--', label=f"{key.replace('_', ' ')}", markersize=5, marker='o', ecolor=colors[n], color=colors[n], capsize=5, capthick=2)
    # Filter out error bars from legend
    handles, labels = ax.get_legend_handles_labels()
    handles = [h[0] if isinstance(h, container.ErrorbarContainer) else h for h in handles]
    ax.legend(handles, labels, fontsize=7, draggable=True, markerscale=0.5, loc='best')

def generate_timing_viewing_plot(timing_data, ax, x_range):
    ax.set_title('Viewing computation time')
    ax.set_xlabel('Swarm count')
    ax.set_ylabel('Computation time (ms)')
    colors = ['purple', 'blue', 'green', 'darkorange', 'black']
    for n, key in enumerate(all_drone_data.keys()):
        data = timing_data[key]
        # Compute average timings
        avg_timing = np.array([np.mean(np.array(data[i])[:, 1], axis=0) for i in range(NB_TESTS)])
        ax.plot(x_range, avg_timing*1000, label=f"{key.replace('_', ' ')}", color=colors[n], linestyle='--', marker='o', markersize=5)
    ax.legend(fontsize='small')

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
        axs[i].set_title(f'Swarm count: {i*5+from_}')
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
            from_ = int(sim_files[0].strip('.npy').split('_')[-1])
            to = int(sim_files[-1].strip('.npy').split('_')[-1])
            steps = int(sim_files[3].strip('.npy').split('_')[-1]) - from_
        except ValueError:
            print('Could not determine range of tests: see file names convention')
            sys.exit(1)
        NB_TESTS = (to - from_)//steps + 1
        print("NB_TESTS: ", NB_TESTS)
        all_drone_data = dict()
        timing_data = dict()
        all_swarm_data = dict()
        params = {}
        try:
            for p, f in zip(filepaths, sim_files):
                data = np.load(p, allow_pickle=True).item()
                w = f.split('_')
                if "topological" in w:
                    wkey = "topological"
                elif "voronoi" in w:
                    wkey = "voronoi"
                elif "vlos" in w:
                    wkey = "vlos"
                else:
                    raise ValueError('Could not determine the neighbors metric')
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
        # Export each plot individually
        x_range = np.arange(from_, to+1)
        elems_title = subfolder.split('_')

        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot(111)
        generate_viewing_error_plot(all_drone_data, ax)
        fig.tight_layout()
        fig.savefig(os.path.join(EXPORT_FOLDER, subfolder, f"viewing_error.png"))

        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot(111)
        generate_coverage_plot(all_swarm_data, ax)
        fig.tight_layout()
        fig.savefig(os.path.join(EXPORT_FOLDER, subfolder, f"coverage.png"))

        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot(111)
        generate_timing_viewing_plot(timing_data, ax, x_range)
        fig.tight_layout()
        fig.savefig(os.path.join(EXPORT_FOLDER, subfolder, f"viewing_timings.png"))

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
        gs = fig.add_gridspec(2,2)
        ax1 = fig.add_subplot(gs[0,0])
        ax2 = fig.add_subplot(gs[0,1])
        ax3 = fig.add_subplot(gs[1,0])
        ax4 = fig.add_subplot(gs[1,1])
        x_range = np.arange(from_, to+1)

        # generate_viewing_error_plot(all_drone_data, ax1)
        generate_avg_num_neighbors_plot(all_drone_data, ax1)
        # generate_coverage_plot(all_swarm_data, ax3)
        # generate_timing_viewing_plot(timing_data, ax4, x_range)
        # generate_swarm_center_plot(all_swarm_data, ax4)

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
