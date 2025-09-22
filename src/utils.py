# src/utils.py

import yaml
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from matplotlib.animation import FuncAnimation

def load_config(config_path="conf/config.yaml"):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    train_r, val_r, test_r = config['data']['train_ratio'], config['data']['val_ratio'], config['data']['test_ratio']
    if not np.isclose(train_r + val_r + test_r, 1.0):
        raise ValueError("Data split ratios in config.yaml must sum to 1.0")
    return config

def location_to_grid(location_str):
    if not isinstance(location_str, str): return None, None
    x_part = ''.join([c for c in location_str if c.isalpha()])
    y_part = ''.join([c for c in location_str if c.isdigit()])
    if not x_part or not y_part: return None, None
    x_coord = ord(x_part.upper()) - ord('A') + 1
    y_coord = int(y_part)
    return x_coord, y_coord

def summarize_metrics(y_true, y_pred, title="Evaluation Metrics"):
    print(f"--- {title} ---")
    print(classification_report(y_true, y_pred, zero_division=0))

def plot_conf_mat(y_true, y_pred, config, title="Confusion Matrix", filename="confusion_matrix.png"):
    plots_dir = config['paths']['plots_dir']
    os.makedirs(plots_dir, exist_ok=True)
    save_path = os.path.join(plots_dir, filename)
    mat = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(20, 16)); sns.heatmap(mat, annot=True, fmt='d', cmap='Blues')
    plt.title(title, fontsize=16); plt.ylabel('Actual'); plt.xlabel('Predicted')
    plt.savefig(save_path, bbox_inches='tight'); plt.show()
    print(f"Confusion matrix saved to {save_path}")

def plot_curves(history, config, title_prefix="", filename="training_curves.png"):
    plots_dir = config['paths']['plots_dir']
    os.makedirs(plots_dir, exist_ok=True)
    save_path = os.path.join(plots_dir, filename)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    ax1.plot(history['train_losses'], label='Training Loss'); ax1.plot(history['val_losses'], label='Validation Loss')
    ax1.set_title('Training and Validation Loss'); ax1.set_xlabel('Epochs'); ax1.legend(); ax1.grid(True)
    ax2.plot(history['val_accuracies'], label='Validation Accuracy', color='green')
    ax2.set_title('Validation Accuracy'); ax2.set_xlabel('Epochs'); ax2.legend(); ax2.grid(True)
    plt.savefig(save_path); plt.show()
    print(f"Training curves saved to {save_path}")

def plot_gradient_descent_progress(tracked_weights, train_losses, config, fold=None):
    plots_dir = config['paths']['plots_dir']
    save_path = os.path.join(plots_dir, f"gradient_progress_fold_{fold}.png")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(tracked_weights, train_losses, marker='o', markersize=4, linestyle='-', color='black')
    ax.scatter(tracked_weights[0], train_losses[0], s=80, color='blue', label='Start', zorder=5)
    ax.scatter(tracked_weights[-1], train_losses[-1], s=80, facecolors='none', edgecolors='red', label='End', zorder=5)
    ax.set_xlabel('Tracked Weight Value (Example)'); ax.set_ylabel('Training Loss')
    ax.set_title(f'Gradient Descent Progress (Fold {fold})'); ax.grid(True, linestyle='--'); ax.legend()
    plt.tight_layout(); plt.savefig(save_path); plt.show()
    print(f"Gradient descent plot saved to {save_path}")

def plot_error_cdf(errors, config, unit="grid units", filename="error_cdf.png"):
    plots_dir = config['paths']['plots_dir']
    save_path = os.path.join(plots_dir, filename)
    e = np.asarray(errors, float); e = e[np.isfinite(e)]; e.sort()
    cdf = np.arange(1, len(e) + 1) / len(e)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.step(e, cdf, where="post"); ax.set_xlim(0, np.quantile(e, 0.99)); ax.set_ylim(0, 1)
    ax.set_xlabel(f"Localization Error ({unit})"); ax.set_ylabel("Cumulative Probability")
    ax.set_title("CDF of Localization Error"); ax.grid(True, linestyle="--")
    mean, e50, e90 = e.mean(), np.quantile(e, 0.50), np.quantile(e, 0.90)
    txt = f"N={len(e)}\nMean={mean:.2f}\nE50={e50:.2f}\nE90={e90:.2f}"
    ax.text(0.98, 0.02, txt, transform=ax.transAxes, ha="right", va="bottom", bbox=dict(boxstyle="round", facecolor="white"))
    plt.tight_layout(); plt.savefig(save_path); plt.show()
    print(f"CDF plot saved to {save_path}")

def animate_paths(true_coords, pred_coords, timestamps, image_path, xrange, yrange, beacon_coords=None, beacon_names=None, title=''):
    fig, ax = plt.subplots(figsize=(10, 8))
    try:
        img = plt.imread(image_path)
        ax.imshow(img, extent=[xrange[0], xrange[1], yrange[0], yrange[1]])
    except FileNotFoundError:
        print(f"Warning: Background image not found at '{image_path}'. Plotting on a blank canvas.")
    
    if beacon_coords:
        for name, (x, y) in beacon_coords.items():
            ax.plot(x, y, '^', markersize=10, color='red')
            ax.text(x + 0.5, y + 0.5, beacon_names.get(name, ""), fontsize=9)
    true_line, = ax.plot([], [], 'g-', label='Actual Path'); pred_line, = ax.plot([], [], 'b--', label='Predicted Path')
    true_dot, = ax.plot([], [], 'go', markersize=8); pred_dot, = ax.plot([], [], 'bo', markersize=8)
    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, ha='left', va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.set_title(title); ax.set_xlabel("X Coordinate"); ax.set_ylabel("Y Coordinate"); ax.legend(); ax.grid(True, alpha=0.6)
    
    def init():
        true_line.set_data([], []); pred_line.set_data([], []); true_dot.set_data([], []); pred_dot.set_data([], []); time_text.set_text('')
        return true_line, pred_line, true_dot, pred_dot, time_text

    def update(frame):
        true_x, true_y = zip(*true_coords[:frame+1]); pred_x, pred_y = zip(*pred_coords[:frame+1])
        true_line.set_data(true_x, true_y); pred_line.set_data(pred_x, pred_y)
        
        true_dot.set_data([true_x[-1]], [true_y[-1]])
        pred_dot.set_data([pred_x[-1]], [pred_y[-1]])

        time_text.set_text(timestamps[frame].strftime('%Y-%m-%d %H:%M:%S'))
        return true_line, pred_line, true_dot, pred_dot, time_text
    
    anim = FuncAnimation(fig, update, frames=len(true_coords), init_func=init, blit=True, interval=100)
    plt.close(fig)
    return anim