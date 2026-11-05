import pandas as pd
import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import json


# file = r'D:\Download\Geolife Trajectories 1.3\Data\174\Trajectory\20080630003333.plt'

def read_plt(file):
    with open(file, 'r') as f:
        lines = f.readlines()
    data_lines = lines[6:]  # Skip the first 6 header lines
    data = []
    for line in data_lines:
        values = line.strip().split(',')
        if len(values) == 7:  # Ensure there are 7 values in each line
            # data.append(values)
            data.append([float(values[0]), float(values[1]), values[5], values[6]])
    return data     
    # df = pd.DataFrame(data, columns=['Latitude', 'Longitude', 'Placeholder', 'Altitude', 'Time', 'date', 'hours'])
    # return df

if __name__ == "__main__":
    person_count = 0
    trajectory_count = 0
    folders = os.listdir(r'D:\Download\Geolife Trajectories 1.3\Data')
    all_trajectories = []
    trajectory_lens = []
    
    for folder in folders:
        folder_path = os.path.join(r'D:\Download\Geolife Trajectories 1.3\Data', folder)
        folder_path = os.path.join(folder_path, 'Trajectory')
        
        if os.path.isdir(folder_path):
            person_count += 1
            files = os.listdir(folder_path)
            trajectory_count += len(files)
            print(f'Processing person {person_count} with {len(files)} trajectories.')
            
            for file in files:
                if file.endswith('.plt'):
                    file_path = os.path.join(folder_path, file)
    
                    dat = read_plt(file_path)
                    all_trajectories.append(dat)
                    trajectory_lens.append(len(dat))
    
    print(f'Total persons: {person_count}')
    print(f'Total trajectories: {trajectory_count}')
    print(f'Example trajectory data (first 5 points): {all_trajectories[0][:5]}')
    print(f'Total trajectories collected: {len(all_trajectories)}')
    
    f = open('all_trajectories.json', 'w')
    json.dump(all_trajectories, f, ensure_ascii=False)
    f.close()
    
    trajectory_lens.sort()
    print(f'Min trajectory length: {trajectory_lens[0]}')
    print(f'Max trajectory length: {trajectory_lens[-1]}')
    print(f'Average trajectory length: {sum(trajectory_lens) / len(trajectory_lens)}')

    