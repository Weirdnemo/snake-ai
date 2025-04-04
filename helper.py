import matplotlib.pyplot as plt
from IPython import display
import numpy as np
import pandas as pd
import os
from datetime import datetime

plt.ion()  # Interactive mode


def plot(scores, mean_scores):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores, label='Game Score')
    plt.plot(mean_scores, label='Mean Score')

    # Add rolling average for smoother visualization
    if len(scores) > 10:
        rolling_avg = pd.Series(scores).rolling(window=10).mean()
        plt.plot(rolling_avg, label='10-Game Rolling Avg', color='green')

    plt.ylim(ymin=0)
    plt.legend()
    plt.text(len(scores) - 1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores) - 1, mean_scores[-1], str(round(mean_scores[-1], 2)))

    # Save the figure periodically
    if len(scores) % 50 == 0:
        save_plot(scores, mean_scores)

    plt.pause(0.1)


def save_plot(scores, mean_scores):
    """Save the current plot to a file"""
    # Create directory if it doesn't exist
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{log_dir}/training_plot_{timestamp}.png"

    plt.figure(figsize=(12, 8))
    plt.title('Snake AI Training Progress')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores, label='Game Score')
    plt.plot(mean_scores, label='Mean Score')

    if len(scores) > 10:
        rolling_avg = pd.Series(scores).rolling(window=10).mean()
        plt.plot(rolling_avg, label='10-Game Rolling Avg', color='green')

    plt.legend()
    plt.savefig(filename)
    print(f"Plot saved to {filename}")


def save_training_data(scores, mean_scores):
    """Save training data to CSV for later analysis"""
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{log_dir}/training_data_{timestamp}.csv"

    data = {
        'game_number': list(range(1, len(scores) + 1)),
        'score': scores,
        'mean_score': mean_scores
    }

    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Training data saved to {filename}")