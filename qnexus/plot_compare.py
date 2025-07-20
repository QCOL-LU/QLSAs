import sys
import os
import json
import pandas as pd
import matplotlib.pyplot as plt


def load_lists_and_label_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    # Expecting 'Errors List' and 'Residuals List' columns as JSON strings
    errors = json.loads(df.iloc[0]['Errors List'])
    residuals = json.loads(df.iloc[0]['Residuals List'])
    backend_label = df.iloc[0]['Backend'] if 'Backend' in df.columns else os.path.basename(csv_path).replace('.csv', '')
    return errors, residuals, backend_label


def plot_comparison(csv_files):
    plt.figure(figsize=(8, 5))
    for csv_file in csv_files:
        errors, _, label = load_lists_and_label_from_csv(csv_file)
        plt.plot(range(len(errors)), [float(e) for e in errors], marker='o', label=label)
    plt.xlabel('IR Iteration')
    plt.ylabel('Error (||x_c - x_q||)')
    plt.yscale('log')
    plt.title('Solution Error vs. Iteration')
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    for csv_file in csv_files:
        _, residuals, label = load_lists_and_label_from_csv(csv_file)
        plt.plot(range(len(residuals)), [float(r) for r in residuals], marker='o', label=label)
    plt.xlabel('IR Iteration')
    plt.ylabel('Residual (||Ax - b||)')
    plt.yscale('log')
    plt.title('Residual Norm vs. Iteration')
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_compare.py <csv_file1> <csv_file2> ...")
        sys.exit(1)
    csv_files = sys.argv[1:]
    for f in csv_files:
        if not os.path.isfile(f):
            print(f"File not found: {f}")
            sys.exit(1)
    plot_comparison(csv_files)


if __name__ == "__main__":
    main()
