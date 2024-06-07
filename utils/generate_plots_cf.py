import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys, os
import cv2
import datetime
from tqdm import tqdm

DATA_FILE_DIR = "./hardware_results"
EXPORT_FOLDER = "."

takeoff_video = None
takeoff_sim = None
out = None
last_plot_img = None

PLOT_LIMITS = [[-2.5, 3.5], [1, 5.5]]
PLOT_SIZE = [4, 4]
PLOT_COLORS = {
    'cf01': 'red',
    'cf04': 'blue',
    'cf05': 'green',
    'cf06': 'orange',
    'cf09': 'purple',
    'cf10': 'brown',
    'cf14': 'black'
}

if __name__ == "__main__":
    # Read arguments
    nb_args = len(sys.argv)
    if nb_args < 2:
        print('Please provide at least one argument --> --visual or --export')
        sys.exit(1)
    
    cmd = sys.argv[1]
    folder = DATA_FILE_DIR
    export_mode = False
    if nb_args == 3:
        folder = os.path.join(DATA_FILE_DIR, sys.argv[2])
    if cmd == '--visual':
        print(f'Visualizing video of folder "{folder}"')
    elif cmd == '--export':
        export_mode = True
        os.makedirs(EXPORT_FOLDER, exist_ok=True)
        print(f'Exporting video of folder "{folder}" to "{EXPORT_FOLDER}"')
    else:
        print('Please provide a valid argument --> --visual or --export')
        sys.exit(1)
    
    # Load the dataset and the video
    files = os.listdir(folder)
    f_csv = [f for f in files if f.endswith('.csv')]
    if "data_trim.csv" in f_csv:
        DATA_FILENNAME = "data_trim.csv"
    else:
        DATA_FILENNAME = f_csv[0]
    print("FOUND DATA FILE: ", DATA_FILENNAME)
    data = pd.read_csv(os.path.join(folder, DATA_FILENNAME))
    f_vid = [f for f in files if f.endswith('.avi') or f.endswith('.mp4')]
    if "video_trim.avi" in f_vid:
        VIDEO_FILENAME = "video_trim.avi"
    else:
        VIDEO_FILENAME = f_vid[0]
    print("FOUND VIDEO FILE: ", VIDEO_FILENAME)

    # Load the video with opencv
    video_ss = cv2.VideoCapture(os.path.join(folder, VIDEO_FILENAME))
    if not video_ss.isOpened():
        print('Error: Cannot open video file')
        sys.exit(1)

    # Extract the video properties
    creation_time = os.path.getctime(os.path.join(folder, VIDEO_FILENAME))
    creation_date = datetime.datetime.fromtimestamp(creation_time)
    fps = video_ss.get(cv2.CAP_PROP_FPS)
    duration = video_ss.get(cv2.CAP_PROP_FRAME_COUNT) / fps
    start_time = creation_date - datetime.timedelta(seconds=duration)
    print(f'Video properties: FPS={fps:.1f}, Duration={duration:.2f}s, Start time={start_time}')
    #print("Video duration: ", duration, "s. Started on ", start_time)

    # Extract first data timestamp
    # if DATA_FILENNAME != "data_trim.csv":
    # data['__time'] = pd.to_datetime(data['__time'], unit='s')
    # print("Data start time: ", data['__time'].iloc[0])
    data_starttime = data['__time'].iloc[0]

    # Create each dataframes for each drone
    dataframes = {}
    for i in [1,4,5,6,9,10,14]:
        key = f'cf{i:02d}'
        columns = [f'/{key}/odom/pose/pose/position/x', f'/{key}/odom/pose/pose/position/y', f'/{key}/odom/pose/pose/position/z', f'/{key}/odom/pose/pose/orientation/yaw']
        dataframes[key] = data[['__time'] + columns].dropna()
        dataframes[key].set_index('__time', inplace=True)
        dataframes[key].index = pd.to_datetime(dataframes[key].index, unit='s')

    # Create unique dataframe with unique timesteps for all drone
    common_timesteps = sorted(set().union(*[dataframes[key].index for key in dataframes.keys()]))

    # Reindex all dataframes
    for key in dataframes.keys():
        dataframes[key] = dataframes[key].reindex(common_timesteps).interpolate()

    # Merge all dataframes into one
    resampling = int(1000000/fps)
    print("Resampling @ ", resampling, " us")
    merged_data = pd.concat(dataframes.values(), axis=1)
    merged_data = merged_data.resample(f'{resampling}U').mean()

    # Named window
    if not export_mode:
        cv2.namedWindow('CF Swarm video', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('CF Swarm video', 1920, 1080)

    # Create animation for all timestamps in dataframe
    #plt.ion()
    fig, ax = plt.subplots(figsize=(PLOT_SIZE[0], PLOT_SIZE[1]))
    fig.tight_layout(pad=1.3)
    takeoff = False
    alive = {key: True for key in dataframes.keys()}
    if export_mode:
        print("EXPORTING VIDEO...")
    for i in tqdm(range(len(merged_data))):
        t = merged_data.index[i]
        # Read the frame
        # video_ss.set(cv2.CAP_PROP_POS_FRAMES, int((t - start_time).total_seconds() * fps))
        ret, frame = video_ss.read()
        if not ret:
            print('WARNING: Cannot read frame')
            continue

        ax.clear()
        ax.set_aspect('equal', adjustable='datalim')
        ax.set_ylim(PLOT_LIMITS[1])
        ax.set_xlim(PLOT_LIMITS[0])
        ax.set_xlabel('Y')
        ax.set_ylabel('X')
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(2)
        ax.tick_params(width=2)
        mean_z = merged_data.loc[t, [f'/{key}/odom/pose/pose/position/z' for key in dataframes.keys()]].mean()
        # ax.set_title(f'Time: {t.strftime("%H:%M:%S.%f")[:-3]} - Swarm altitude: {mean_z:.2f} m')
        ax.set_title(f'Swarm altitude: {mean_z:.2f} m')
        data_ok = True
        for key in dataframes.keys():
            x, y, z, yaw = merged_data.loc[t, [f'/{key}/odom/pose/pose/position/x', f'/{key}/odom/pose/pose/position/y', f'/{key}/odom/pose/pose/position/z', f'/{key}/odom/pose/pose/orientation/yaw']]
            if np.isnan(x) or np.isnan(y) or np.isnan(z) or np.isnan(yaw):
                data_ok = False
            # Only plot in air drones
            if z > 0.3 and not takeoff:
                takeoff = True
                takeoff_sim = t
                if not export_mode:
                    print(f'Takeoff at {t.strftime("%H:%M:%S.%f")[:-1]}')
            if takeoff and z < 0.1:
                alive[key] = False
            if alive[key] or not takeoff:
                ax.plot(-y, x, 'o', label=key, c=PLOT_COLORS[key], markersize=10)
                # Add arrow for orientation
                ax.arrow(-y, x, -0.2*np.sin(yaw), 0.2*np.cos(yaw), head_width=0.07, head_length=0.12, fc=PLOT_COLORS[key], ec=PLOT_COLORS[key])

        # Convert plot to image
        fig.canvas.draw()
        plot_img = np.array(fig.canvas.buffer_rgba())[:,:,:3]
        plot_img = plot_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        if not data_ok and last_plot_img is not None:
            plot_img = last_plot_img
        else:
            last_plot_img = plot_img

        # Overlay plot on video frame
        frame[:plot_img.shape[0], :plot_img.shape[1]] = cv2.addWeighted(frame[:plot_img.shape[0], :plot_img.shape[1]], 0.3, plot_img, 0.7, 0)

        if export_mode:
            if out is None:
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out = cv2.VideoWriter(os.path.join(folder, 'video_final.avi'), fourcc, fps, (frame.shape[1], frame.shape[0]))
            else:
                out.write(frame)
        else:
            # Display combined image
            cv2.imshow('CF Swarm video', frame)

            key = cv2.waitKey(1)
            if key == ord('q'):
                print("GOODBYE!")
                break
            elif key == ord('k'):
                # Readjust video start time and export it
                takeoff_video = t
                print(f'Takeoff in video at {takeoff_video.strftime("%H:%M:%S.%f")[:-1]}')
            if takeoff_video is not None and takeoff_sim is not None:
                # If video takeoff is before simulation takeoff, trim data
                if (takeoff_video - takeoff_sim).total_seconds() < 0:
                    # Export df with truncated time
                    start = data_starttime  + (takeoff_sim - takeoff_video).total_seconds()
                    # Save dataframe
                    data[data.__time >= start].to_csv(os.path.join(folder, 'data_trim.csv'))
                    print('Data exported!')
                    break
                else:
                    # If video takeoff is after simulation takeoff, trim video
                    diff = (takeoff_video - takeoff_sim).total_seconds()
                    video_ss.set(cv2.CAP_PROP_POS_FRAMES, int(diff * fps))
                    print(f'Difference between takeoff: {diff:.2f}')
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    out = cv2.VideoWriter(os.path.join(folder, f'video_trim.avi'), fourcc, fps, (frame.shape[1], frame.shape[0]))
                    nb_frames = int((duration - diff) * fps)
                    print("EXPORTING TRIMMED VIDEO...")
                    # Add progress bar
                    for _ in tqdm(range(nb_frames)):
                        ret, frame = video_ss.read()
                        if not ret:
                            raise Exception('Cannot read frame')
                        out.write(frame)
                    out.release()
                    print('Video exported!') 
                    break

    # Release video capture
    if export_mode:
        out.release()
        print("Video exported! --> ", os.path.join(os.path.join(folder, 'video_final.avi')))

