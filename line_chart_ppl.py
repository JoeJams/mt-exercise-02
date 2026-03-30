# Exercise Task:
# Create a line chart each for the training and the validation perplexity to visualize the
# results. Write and commit a python script for creating the tables and line plots, taking
# the log files from the previous step as input

import matplotlib.pyplot as plt
import pandas as pd
import os
import argparse

# Read in all the plots in a first step and store them in one dataframe for easier plotting (in the same structure as on the exercise)
# File Structure of the log files:
# I.e. one dataframe for each perplexity type (training, validation and test)
# Files are csv format with columns "Epoch", "Training Perplexity", "Validation Perplexity"
# The last line of the file contains the final test perplexity
# The dropout rate is specified in the first line of the file (e.g. "Perplexity Log for Dropout Rate 0.5")


def read_perp_logs(log_dir):
    '''
    Read in all the log files in the specified directory and extract the training, validation and test perplexities for each dropout rate.
    Args:
        log_dir (str): Directory where the log files are located.
    Returns:
        training_ppls (dict): Dictionary with dropout rates as keys and lists of training perplexities as values.
        val_ppls (dict): Dictionary with dropout rates as keys and lists of validation perplexities
        test_ppls (dict): Dictionary with dropout rates as keys and final test perplexities as values.
    '''
    training_ppls = {}
    val_ppls = {}
    test_ppls = {}

    for filename in os.listdir(log_dir):
        if filename.startswith('perplexity_log_d') and filename.endswith('.csv'):
            with open(os.path.join(log_dir, filename), 'r') as f: # read the whole file except the first line and the last line
                first_line = f.readline()
                dropout_rate = first_line.split('Dropout Rate ')[1].split()[0]
                # Read the final test perplexity from the last line of the file
                test_line = f.readlines()[-1]
                test_ppls[dropout_rate] = float(test_line.split('Final Test Perplexity: ')[1].strip())
                # Read the rest of the file into a dataframe (skipping the first line and the last line)
                df = pd.read_csv(os.path.join(log_dir, filename), skiprows=1, skipfooter=1, engine='python')
                training_ppls[dropout_rate] = df['Training Perplexity'].tolist()
                val_ppls[dropout_rate] = df['Validation Perplexity'].tolist()

    return training_ppls, val_ppls, test_ppls

def create_tables(training_ppls, val_ppls, test_ppls):
    '''
    Create tables for training, validation and test perplexities and save them as csv files.
    Args:
        training_ppls (dict): Dictionary with dropout rates as keys and lists of training perplexities as values.
        val_ppls (dict): Dictionary with dropout rates as keys and lists of validation perplexities as values.
        test_ppls (dict): Dictionary with dropout rates as keys and final test perplexities as values.
    Returns:
        None
    '''

    # Check if we have data to process
    if not training_ppls or not val_ppls:
        print("No perplexity data found. Ensure log files are in the correct format.")
        return
    
    with open('training_perplexity.csv', 'w') as f:
        f.write('Epoch,' + ','.join(training_ppls.keys()) + '\n')
        for epoch in range(1, len(next(iter(training_ppls.values()))) + 1):
            f.write(f'{epoch},' + ','.join(f'{training_ppls[dropout][epoch-1]:.4f}' for dropout in training_ppls.keys()) + '\n')

    with open('validation_perplexity.csv', 'w') as f:
        f.write('Epoch,' + ','.join(val_ppls.keys()) + '\n')
        for epoch in range(1, len(next(iter(val_ppls.values()))) + 1):
            f.write(f'{epoch},' + ','.join(f'{val_ppls[dropout][epoch-1]:.4f}' for dropout in val_ppls.keys()) + '\n')

    with open('test_perplexity.csv', 'w') as f:
        f.write('Test Perplexity,' + ','.join(test_ppls.keys()) + '\n')
        f.write(',' + ','.join(f'{test_ppls[dropout]:.4f}' for dropout in test_ppls.keys()) + '\n')
    
    print("Tables created successfully: training_perplexity.csv, validation_perplexity.csv, test_perplexity.csv")

def plot_perplexities(training_ppls, val_ppls, save_plots=True, show_plots=False):
    '''
    Create line plots for training and validation perplexities over epochs for different dropout rates.
    Args:
        training_ppls (dict): Dictionary with dropout rates as keys and lists of training perplexities
        val_ppls (dict): Dictionary with dropout rates as keys and lists of validation perplexities as values.
        save_plots (bool): Flag to save the plots as png files.
        show_plots (bool): Flag to display the plots after creation.
    Returns:
        None
    '''
    # Create two separate plots for training and validation perplexities
    plt.figure(figsize=(12, 6))
    for dropout, training_ppl in training_ppls.items():
        plt.plot(range(1, len(training_ppl) + 1), training_ppl, label=f'Training Perplexity (d={dropout})')
    plt.xlabel('Epoch')
    plt.ylabel('Perplexity') 
    plt.title('Training Perplexity over Epochs for Different Dropout Rates')
    plt.legend()
    if save_plots:
        plt.savefig('training_perplexity.png')
    if show_plots:
        plt.show()
    plt.figure(figsize=(12, 6))
    for dropout, val_ppl in val_ppls.items():
        plt.plot(range(1, len(val_ppl) + 1), val_ppl, label=f'Validation Perplexity (d={dropout})')
    plt.xlabel('Epoch')
    plt.ylabel('Perplexity')    
    plt.title('Validation Perplexity over Epochs for Different Dropout Rates')
    plt.legend()
    if save_plots:
        plt.savefig('validation_perplexity.png')
    if show_plots:
        plt.show()

# Argparse for the script to specify the directory where the log files are located (and some additional flags for easier handling of the output)
def parse_args():
    parser = argparse.ArgumentParser(description='Plot training and validation perplexities from log files')
    parser.add_argument('--log_dir', type=str, default='tools/pytorch-examples/word_language_model', help='Directory where the log files are located')
    parser.add_argument('--save_tables', action='store_true', help='Flag to save the tables as csv files')
    parser.add_argument('--save_plots', action='store_true', help='Flag to save the plots as png files')
    parser.add_argument('--show_plots', action='store_true', help='Flag to display the plots after creation')
    parser.add_argument('--output_dir', type=str, default='.', help='Directory to save the output tables and plots')
    return parser.parse_args()

def main():
    args = parse_args()
    training_ppls, val_ppls, test_ppls = read_perp_logs(args.log_dir)
    if args.save_tables:
        create_tables(training_ppls, val_ppls, test_ppls)
    if args.save_plots:
        plot_perplexities(training_ppls, val_ppls, save_plots=True, show_plots=args.show_plots)
    if args.show_plots:
        plot_perplexities(training_ppls, val_ppls, save_plots=False, show_plots=True)

if __name__ == "__main__":
    main()