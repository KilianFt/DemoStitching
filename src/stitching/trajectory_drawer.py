#!/usr/bin/env python3
"""
Interactive Trajectory Drawing, Saving, and Loading Tool

This script allows users to:
1. Draw trajectories interactively using matplotlib
2. Save drawn trajectories to files
3. Load previously saved trajectories
4. Add noise to trajectories
5. Generate multiple demonstrations from base trajectories

Usage:
    python trajectory_drawer.py
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from typing import List, Tuple, Optional
import argparse


class TrajectoryDrawer:
    """Interactive trajectory drawing and management tool."""
    
    def __init__(self, canvas_size=(15, 15)):
        self.trajectories = []  # List of trajectory sets
        self.current_trajectory = []  # Points being drawn
        self.drawing = False
        self.fig = None
        self.ax = None
        self.canvas_size = canvas_size
        self.current_line = None  # Line object for current trajectory being drawn
        self.trajectory_lines = []  # Store line objects for all trajectories
        
    def start_interactive_drawing(self):
        """Start interactive drawing session."""
        print("Interactive Trajectory Drawing")
        print("Instructions:")
        print("- Click and drag to draw trajectories")
        print("- Press 'n' to start a new trajectory")
        print("- Press 's' to save current trajectories")
        print("- Press 'c' to clear all trajectories")
        print("- Press 'r' to resize canvas")
        print("- Press 'q' to quit")
        print("- Close window to finish drawing")
        print(f"- Current canvas size: {self.canvas_size[0]} x {self.canvas_size[1]}")
        
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.ax.set_xlim(0, self.canvas_size[0])
        self.ax.set_ylim(0, self.canvas_size[1])
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_title(f'Draw Trajectories - Canvas: {self.canvas_size[0]}x{self.canvas_size[1]}')
        self.ax.grid(True, alpha=0.3)
        
        # Enable interactive mode for real-time updates
        plt.ion()
        
        # Connect event handlers
        self.fig.canvas.mpl_connect('button_press_event', self._on_press)
        self.fig.canvas.mpl_connect('button_release_event', self._on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self._on_motion)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        
        # Show the plot
        plt.show()
        
        # Keep the window open and responsive
        try:
            # This keeps the window open and responsive to events
            while plt.get_fignums():
                plt.pause(0.01)
        except KeyboardInterrupt:
            print("\nDrawing session interrupted.")
        finally:
            plt.ioff()
        
    def _on_press(self, event):
        """Handle mouse press events."""
        if event.inaxes != self.ax:
            return
        self.drawing = True
        self.current_trajectory = [(event.xdata, event.ydata)]
        
        # Initialize current line for real-time drawing
        self.current_line, = self.ax.plot([], [], 'r-', linewidth=2, alpha=0.8)
        plt.draw()
        plt.pause(0.001)
        
    def _on_release(self, event):
        """Handle mouse release events."""
        if not self.drawing:
            return
        self.drawing = False
        
        if len(self.current_trajectory) > 1:
            # Convert to numpy array and add to trajectories
            traj_array = np.array(self.current_trajectory)
            self.trajectories.append(traj_array)
            
            # Remove the temporary red line
            if self.current_line:
                self.current_line.remove()
            
            # Plot the final trajectory in blue with start/end markers
            line, = self.ax.plot(traj_array[:, 0], traj_array[:, 1], 'b-', linewidth=2, alpha=0.7)
            start_marker, = self.ax.plot(traj_array[0, 0], traj_array[0, 1], 'go', markersize=8)
            end_marker, = self.ax.plot(traj_array[-1, 0], traj_array[-1, 1], 'ro', markersize=8)
            
            # Store line objects for later manipulation
            self.trajectory_lines.append((line, start_marker, end_marker))
            
            plt.draw()
            plt.pause(0.001)
            
            print(f"Trajectory {len(self.trajectories)} added with {len(traj_array)} points")
        else:
            # Remove the temporary line if trajectory was too short
            if self.current_line:
                self.current_line.remove()
                plt.draw()
                plt.pause(0.001)
        
        self.current_trajectory = []
        self.current_line = None
        
    def _on_motion(self, event):
        """Handle mouse motion events."""
        if not self.drawing or event.inaxes != self.ax:
            return
        
        self.current_trajectory.append((event.xdata, event.ydata))
        
        # Update the current line in real-time
        if len(self.current_trajectory) > 1 and self.current_line:
            traj_array = np.array(self.current_trajectory)
            self.current_line.set_data(traj_array[:, 0], traj_array[:, 1])
            plt.draw()
            plt.pause(0.001)
            
    def _on_key(self, event):
        """Handle key press events."""
        if event.key == 'n':
            print("Starting new trajectory...")
        elif event.key == 's':
            filename = input("Enter filename to save (without extension): ")
            if filename:
                self.save_trajectories(f"{filename}.pkl")
        elif event.key == 'c':
            self.clear_trajectories()
        elif event.key == 'r':
            self.resize_canvas()
        elif event.key == 'q':
            plt.close()
            
    def clear_trajectories(self):
        """Clear all drawn trajectories."""
        self.trajectories = []
        self.trajectory_lines = []
        self.ax.clear()
        self.ax.set_xlim(0, self.canvas_size[0])
        self.ax.set_ylim(0, self.canvas_size[1])
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_title(f'Draw Trajectories - Canvas: {self.canvas_size[0]}x{self.canvas_size[1]}')
        self.ax.grid(True, alpha=0.3)
        plt.draw()
        plt.pause(0.001)
        print("All trajectories cleared")
        
    def resize_canvas(self):
        """Resize the drawing canvas."""
        try:
            width = float(input(f"Enter canvas width (current: {self.canvas_size[0]}): ") or self.canvas_size[0])
            height = float(input(f"Enter canvas height (current: {self.canvas_size[1]}): ") or self.canvas_size[1])
            
            self.canvas_size = (width, height)
            self.ax.set_xlim(0, width)
            self.ax.set_ylim(0, height)
            self.ax.set_title(f'Draw Trajectories - Canvas: {width}x{height}')
            plt.draw()
            plt.pause(0.001)
            print(f"Canvas resized to {width} x {height}")
        except ValueError:
            print("Invalid input. Canvas size unchanged.")
        
    def save_trajectories(self, filename: str):
        """Save trajectories to file."""
        if not self.trajectories:
            print("No trajectories to save!")
            return
            
        # Calculate velocities for each trajectory
        trajectory_data = {
            'trajectories': [],
            'velocities': [],
            'metadata': {
                'n_trajectories': len(self.trajectories),
                'creation_time': np.datetime64('now')
            }
        }
        
        for traj in self.trajectories:
            # Ensure minimum number of points for velocity calculation
            if len(traj) < 2:
                continue
                
            trajectory_data['trajectories'].append(traj)
            # Calculate velocity using gradient (same as in load_tools.py)
            velocity = np.gradient(traj, axis=0)
            trajectory_data['velocities'].append(velocity)
        
        try:
            with open(filename, 'wb') as f:
                pickle.dump(trajectory_data, f)
            print(f"Saved {len(trajectory_data['trajectories'])} trajectories to {filename}")
        except Exception as e:
            print(f"Error saving trajectories: {e}")
            
    def load_trajectories(self, filename: str) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Load trajectories from file."""
        try:
            with open(filename, 'rb') as f:
                data = pickle.load(f)
            
            trajectories = data['trajectories']
            velocities = data['velocities']
            
            print(f"Loaded {len(trajectories)} trajectories from {filename}")
            return trajectories, velocities
            
        except Exception as e:
            print(f"Error loading trajectories: {e}")
            return [], []
            
    def add_noise_to_trajectories(self, trajectories: List[np.ndarray], 
                                noise_std: float = 0.2) -> List[np.ndarray]:
        """Add Gaussian noise to trajectories."""
        noisy_trajectories = []
        
        for traj in trajectories:
            noise = np.random.normal(0, noise_std, traj.shape)
            noisy_traj = traj + noise
            noisy_trajectories.append(noisy_traj)
            
        print(f"Added noise (std={noise_std}) to {len(trajectories)} trajectories")
        return noisy_trajectories
    
    def plot_trajectory_set(self, trajectory_sets: List[List[np.ndarray]], 
                          title: str = "Multiple Trajectory Sets"):
        """Plot multiple sets of trajectories."""
        fig, axes = plt.subplots(1, len(trajectory_sets), figsize=(5*len(trajectory_sets), 5))
        
        if len(trajectory_sets) == 1:
            axes = [axes]
            
        for set_idx, traj_set in enumerate(trajectory_sets):
            ax = axes[set_idx]
            
            for traj in traj_set:
                ax.plot(traj[:, 0], traj[:, 1], 'b-', alpha=0.5, linewidth=1)
                ax.plot(traj[0, 0], traj[0, 1], 'go', markersize=4)
                ax.plot(traj[-1, 0], traj[-1, 1], 'ro', markersize=4)
                
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_title(f'Set {set_idx + 1} ({len(traj_set)} demos)')
            ax.grid(True, alpha=0.3)
            
        plt.tight_layout()
        plt.waitforbuttonpress()
        plt.close()


def plot_trajectories(trajectories: List[np.ndarray], title: str = "Trajectories"):
    """Plot trajectories."""
    plt.figure(figsize=(10, 8))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(trajectories)))
    
    for i, traj in enumerate(trajectories):
        plt.plot(traj[:, 0], traj[:, 1], color=colors[i], linewidth=2, alpha=0.7, label=f'Traj {i+1}')
        plt.plot(traj[0, 0], traj[0, 1], 'go', markersize=8)
        plt.plot(traj[-1, 0], traj[-1, 1], 'ro', markersize=8)
        
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    # plt.legend()
    plt.waitforbuttonpress()
    plt.close()


def main():
    from src.util.load_tools import generate_multiple_demos
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description='Interactive Trajectory Drawing Tool')
    parser.add_argument('--load', type=str, help='Load trajectories from file')
    parser.add_argument('--noise', type=float, default=0.2, help='Noise standard deviation')
    parser.add_argument('--demos', type=int, default=5, help='Number of demonstrations to generate')
    parser.add_argument('--plot-only', action='store_true', help='Only plot loaded trajectories')
    parser.add_argument('--canvas-width', type=float, default=15, help='Canvas width (default: 15)')
    parser.add_argument('--canvas-height', type=float, default=15, help='Canvas height (default: 15)')
    
    args = parser.parse_args()
    
    drawer = TrajectoryDrawer(canvas_size=(args.canvas_width, args.canvas_height))
    
    if args.load:
        # Load and process existing trajectories
        trajectories, velocities = drawer.load_trajectories(args.load)
        
        if trajectories:
            if args.plot_only:
                plot_trajectories(trajectories, "Loaded Trajectories")
            else:
                # Generate multiple demos with noise
                demo_sets = generate_multiple_demos(trajectories, args.demos, args.noise)
                drawer.plot_trajectory_set(demo_sets, "Generated Demonstrations")
                
                # Save the generated demos
                save_name = args.load.replace('.pkl', f'_demos_{args.demos}.pkl')
                demo_data = {
                    'trajectory_sets': demo_sets,
                    'base_trajectories': trajectories,
                    'metadata': {
                        'n_demos': args.demos,
                        'noise_std': args.noise,
                        'generation_time': np.datetime64('now')
                    }
                }
                
                with open(save_name, 'wb') as f:
                    pickle.dump(demo_data, f)
                print(f"Saved generated demonstrations to {save_name}")
    else:
        # Start interactive drawing
        drawer.start_interactive_drawing()


if __name__ == "__main__":
    main()
