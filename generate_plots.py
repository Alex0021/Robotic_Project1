import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
from matplotlib.colors import Normalize, ListedColormap
import sys, os

ROOT_FOLDER = 'sim_results'
EXPORT_FOLDER =  'exported_plots'

def generate_viewing_error_plot(all_drone_data, ax, x_range, y_range):
    ax.set_title('Drones viewing direction error')
    # Labelling
    ax.set_xlabel('# neighbors')
    ax.set_ylabel('Drone index')
    ax.set_zlabel('Error')
    # Create the bins
    xpos, ypos = np.meshgrid(x_range, y_range)
    xpos = xpos.flatten('F')
    ypos = ypos.flatten('F')
    zpos = np.zeros_like(xpos)
    # Find the average error for each drone
    errors = np.mean(all_drone_data[:, :, :, 3], axis=1)
    dist_to_hull = np.mean(np.abs(all_drone_data[:, :, :, 2]), axis=1)
    dist_sorted_indices = np.argsort(dist_to_hull, axis=1)
    for idx, i in zip(dist_sorted_indices, range(len(dist_sorted_indices))):
        errors[i,:] = errors[i, idx]
        dist_to_hull[i,:] = dist_to_hull[i, idx]


    # Construct arrays with the dimensions for the bars.
    dx = 0.5 * np.ones_like(xpos)
    dy = 0.5 * np.ones_like(ypos)
    dz = errors.flatten()

    # Create color map with dist to center (on average)
    colormap = mpl.colormaps['viridis']
    new_cm = ListedColormap(colormap(np.linspace(0.25, 0.75, 256)))
    #max_height = np.max(dist_to_hull)
    #min_height = np.min(dist_to_hull)
    dist_to_hull = dist_to_hull.flatten()
    #colors = new_cm((dist_to_hull - min_height) / (max_height-min_height))
    colors = new_cm(dist_to_hull)
    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors, zsort='average')
    #ax.set_box_aspect(aspect=None, zoom=0.95)
    cbar = plt.colorbar(cm.ScalarMappable(Normalize(vmin=0, vmax=1), cmap=new_cm), ax=ax, orientation='vertical', fraction=0.025, pad=0.04)
    cbar.set_label('Distance to convex hull center')

    # Add average over all drones as 2d line
    avg_errors = np.mean(errors, axis=1)
    ax.plot(x_range, avg_errors, zs=10, zdir='y', color='black', label='Avg error', linewidth=2)
    ax.legend()

def generate_coverage_plot(all_drone_data, ax):
    ax.set_title('Mean and std of the coverage')
    ax.set_xlabel('# neighbors')
    ax.set_ylabel('Coverage %')
    # Find the average coverage for each drone
    coverage_mean = np.mean(all_drone_data[:, :, 0, 4], axis=1)
    coverage_std = np.std(all_drone_data[:, :, 0, 4], axis=1)
    ax.errorbar(np.arange(from_, to+1), coverage_mean*100, yerr=[coverage_std*100, np.clip(coverage_std, 0, 1-coverage_mean)*100], fmt='k--', label='Coverage', markersize=10, marker='o', ecolor='red', capsize=5, capthick=2)
    ax.legend()

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
        value = round(timings_neighborhood[i]/timings_total[i] ,2)
        ax.text(x_range[i], timings_neighborhood[i]/2, value, ha = 'center')
        value = round(timings_viewing[i]/timings_total[i] ,2)
        ax.text(x_range[i], timings_neighborhood[i]+timings_viewing[i]/2, value, ha = 'center')
        value = round(timings_coverage[i]/timings_total[i] ,2)
        ax.text(x_range[i], timings_neighborhood[i]+timings_viewing[i]+timings_coverage[i]/2, value, ha = 'center')
    ax.legend()

def generate_swarm_center_plot(swarm_centers, ax):
    ax.set_title('Swarm center position')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    # Compute center x,y
    center_x = swarm_centers[:, :, 0]
    center_y = swarm_centers[:, :, 1]
    for i in range(center_x.shape[0]):
        ax.plot(center_x[i, :], center_y[i, :], label=f'Run {i+1}')
    # ax.plot(center_x, center_y, 'b-', linewidth=2)
    ax.grid(True)
    ax.legend()

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
        subfolders = [sys.argv[2]]
    
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
        except ValueError:
            print('Could not determine range of tests: see file names convention')
            sys.exit(1)
        NB_TESTS = to - from_ + 1
        data = []
        timing_data = []
        try:
            for p in filepaths:
                data.append(np.load(p, allow_pickle=True).item())
        except FileNotFoundError:
            print('File not found')

        min_data_len = np.min([len(d['drone_data']) for d in data])
        all_drone_data = np.array([d['drone_data'][:min_data_len] for d in data])
        timings = np.array([d['timings'][:min_data_len] for d in data])
        swarm_centers = np.array([d['swarm_center'][:min_data_len] for d in data])
        nb_drones = data[0]['params']['drone_count']

        # Create the plots
        fig = plt.figure(figsize=(20, 12))
        elems_title = sim_files[0].split('_')
        fig.suptitle(f"Results for testing {' '.join(elems_title[:-1])} metric", fontsize=16)
        gs = fig.add_gridspec(3,5)
        ax1 = fig.add_subplot(gs[0:3,0:3], projection='3d')
        ax2 = fig.add_subplot(gs[0,3:5])
        ax3 = fig.add_subplot(gs[1,3:5])
        ax4 = fig.add_subplot(gs[2,3:5])
        x_range = np.arange(from_, to+1)
        y_range = np.arange(nb_drones)

        generate_viewing_error_plot(all_drone_data, ax1, x_range, y_range)
        generate_coverage_plot(all_drone_data, ax2)
        generate_timings_plot(timings, ax3, x_range)
        generate_swarm_center_plot(swarm_centers, ax4)

        if not export_mode and nb_args == 3:
            # Create metrics summary
            swarm_spread = data[0]['params']['swarm_spread']
            r_coh = data[0]['params']['r_coh']
            neighbors_metric = data[0]['params']['neighbors']['metric']
            neighbors_count = data[0]['params']['neighbors']['count']
            neighbors_range = data[0]['params']['neighbors']['sensing_range']
            neighbors_r_agent = data[0]['params']['neighbors']['r_agent']
            noise = data[0]['params']['noise']
            viewing_dim = '2D' if data[0]['params']['viewing_metric']['dim'] == 2 else '3D'
            fig2, ax = plt.subplots(1, 1)
            ax.set_title('Metrics summary', fontweight='bold', fontstyle='oblique', fontsize=16)
            ax.axis('off')
            ax.text(0, 0.9, f'Number of drones: {nb_drones}', fontsize=12)
            ax.text(0, 0.8, f'Swarm spread: {swarm_spread:.2f}', fontsize=12)
            ax.text(0, 0.7, f'Cohesion radius: {r_coh:.2f}', fontsize=12)
            ax.text(0, 0.6, f'Neighbors:  ', fontsize=12)
            ax.text(0.25, 0.6, f'Metric: {neighbors_metric}', fontsize=12)
            ax.text(0.25, 0.53, f'Count: {neighbors_count}', fontsize=12)
            ax.text(0.25, 0.46, f'Sensing range: {neighbors_range:.2f}', fontsize=12)
            ax.text(0.25, 0.39, f'Agent radius: {neighbors_r_agent:.3f}', fontsize=12)
            ax.text(0, 0.3, f'Noise: ', fontsize=12)
            ax.text(0.25, 0.3, f'type: {noise["type"]}', fontsize=12)
            ax.text(0.25, 0.23, f'dist: {noise["param_dist"]:.2f}', fontsize=12)
            ax.text(0.25, 0.16, f'cone: {noise["param_dir"]:.2f}', fontsize=12)
            ax.text(0, 0.05, f'Viewing metric dim: {viewing_dim}', fontsize=12)
            fig2.tight_layout()
        
        fig.tight_layout()
        if export_mode:
            fig.savefig(os.path.join(EXPORT_FOLDER, f'{subfolder}.svg'))

    print('\nDONE GENERATING PLOTS!')
    if export_mode:
        print(f'Plots have been exported to "{EXPORT_FOLDER}"!')
    else:
        plt.show()
