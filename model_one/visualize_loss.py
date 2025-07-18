import os
import json
import matplotlib.pyplot as plt

def load_eval_records(eval_file_path):
    """
    Load the evaluation records from a JSON file.
    Each record should be a dict with 'step' and 'eval_loss'.
    """
    if not os.path.exists(eval_file_path):
        raise FileNotFoundError(f"Eval file not found: {eval_file_path}")
    with open(eval_file_path, 'r', encoding='utf-8') as f:
        records = json.load(f)
    return records


def plot_eval_losses(records, output_path=None):
    """
    Plot eval_loss vs. training step.

    Args:
        records (List[Dict]): List of {'step': int, 'eval_loss': float}.
        output_path (str, optional): If provided, save the plot to this path.
    """
    # Extract data
    steps = [r['step'] for r in records]
    losses = [r['eval_loss'] for r in records]

    # Create plot
    plt.figure()
    plt.plot(steps, losses, marker='o', linestyle='-')
    plt.title('Evaluation Loss over Training Steps')
    plt.xlabel('Global Step')
    plt.ylabel('Eval Loss')
    plt.grid(True)

    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    else:
        plt.show()


def main():
    # Adjust these paths as needed
    eval_json = os.path.join('sft_llm_fsdp', 'eval_losses.json')
    output_png = os.path.join('sft_llm_fsdp', 'eval_loss_plot.png')

    records = load_eval_records(eval_json)
    plot_eval_losses(records, output_png)

if __name__ == '__main__':
    main()
