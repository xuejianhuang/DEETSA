import os
import pandas as pd
import matplotlib.pyplot as plt


def plot_loss(csv_files, save_path=None, dpi=300):
    plt.figure(figsize=(8, 6))

    for file in csv_files:
        # Read the CSV file
        data = pd.read_csv(file)

        # Assume the CSV file contains 'Step' and 'Value' columns
        if 'Step' not in data.columns or 'Value' not in data.columns:
            print(f"File {file} is incorrectly formatted. Please ensure it contains 'Step' and 'Value' columns.")
            continue

        # Plot the loss curve
        plt.plot(data['Step'], data['Value'], label=os.path.basename(file)[0:-4], linewidth=2)

    # Set labels and adjust font sizes
    plt.xlabel('Iterations', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    # plt.title('Training Loss Visualization', fontsize=16)

    # Set legend and adjust font size
    plt.legend(fontsize=14)
    # plt.legend(prop={'size': 22})

    plt.legend()
    # plt.grid(True)

    # Save the plot if a save path is specified
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Image saved as {save_path}")

    plt.show()


base = "twitter"
# Example usage
csv_files = [
    f'{base}/lr/Learning Rate=5e-4.csv',  # Replace with actual path
    f'{base}/lr/Learning Rate=5e-5.csv',  # Replace with actual path
    f'{base}/lr/Learning Rate=5e-6.csv'
    # More files...
]

if __name__ == '__main__':
    # Call the function and save as a PNG file
    save_path = f'{base}/lr/loss_plot.png'  # Replace with your desired save path
    plot_loss(csv_files, save_path=save_path, dpi=600)

    plot_loss(csv_files)
