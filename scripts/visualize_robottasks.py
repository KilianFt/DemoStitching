import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_task(file_path, output_dir):
    """Visualizes a single robot task from an npy file."""
    task_name = os.path.basename(file_path).replace('.npy', '')
    data = np.load(file_path)
    
    # data.shape is (9, 1000, 7)
    # 9 demonstrations, 1000 time steps, 7 values (x, y, z, qx, qy, qz, qw)
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    colors = plt.cm.viridis(np.linspace(0, 1, data.shape[0]))
    
    for i in range(data.shape[0]):
        x = data[i, :, 0]
        y = data[i, :, 1]
        z = data[i, :, 2]
        
        ax.plot(x, y, z, color=colors[i], alpha=0.7, label=f'Demo {i+1}' if i == 0 else "")
        # Mark start and end
        ax.scatter(x[0], y[0], z[0], color='g', marker='o', s=50)
        ax.scatter(x[-1], y[-1], z[-1], color='r', marker='x', s=50)

    ax.set_title(f'Robot Task: {task_name}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'{task_name}.png')
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved visualization for {task_name} to {output_path}")

def main():
    root_dir = os.getcwd()
    dataset_dir = os.path.join(root_dir, 'dataset', 'robottasks', 'pos_ori')
    output_dir = os.path.join(root_dir, 'dataset', 'robottasks', 'plots')
    
    if not os.path.exists(dataset_dir):
        print(f"Dataset directory not found: {dataset_dir}")
        return

    npy_files = [f for f in os.listdir(dataset_dir) if f.endswith('.npy')]
    
    if not npy_files:
        print(f"No .npy files found in {dataset_dir}")
        return

    for npy_file in npy_files:
        file_path = os.path.join(dataset_dir, npy_file)
        visualize_task(file_path, output_dir)

if __name__ == "__main__":
    main()
