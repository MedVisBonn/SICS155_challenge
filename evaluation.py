import os
import click
from sklearn.metrics import f1_score, accuracy_score, average_precision_score
from utils import read_phase_file, load_action_map, edit_score 


@click.command()
@click.argument('groundtruth_path', type=click.Path(exists=True, file_okay=False))
@click.argument('prediction_path', type=click.Path(exists=True, file_okay=False))
@click.option('--mapping_path', type=click.Path(), defaault='mapping.txt')
@click.option('--probability_path', type=click.Path(), default=None)
def evaluate(groundtruth_path, prediction_path, mapping_path, probability_path):
    """
    Evaluate the performance of a phase recognition algorithm. Ground truth and prediction files should have the same file names.

    GROUNDTRUTH_PATH: Path to the folder containing ground truth phase files.
    PREDICTION_PATH: Path to the folder containing prediction phase files.
    mapping_path: Path to the mapping file for mapping phase names to integer (optional, default=mapping.txt).
    probability_path: To calculate the Area under the curve (AUC) for the probability of each phase, a probability file in the numpy format ".npy" is required (optional).
    """
    ground_truth_files = sorted(os.listdir(groundtruth_path))
    #prediction_files = sorted(os.listdir(prediction_path))
    #probability_files = sorted(os.listdir(probability_path)) if probability_path else []
    phase_map = load_action_map(mapping_path)

    metrics = {
            'accuracy': [],
            'f1_score': [],
            'edit_score': [],
            'pr_auc': []
    }

    for gt_file in ground_truth_files:
        gt_path = os.path.join(groundtruth_path, gt_file)
        pred_path = os.path.join(prediction_path, gt_file)
        probs_path = os.path.join(probability_path, os.path.splitext(gt_file)[0] + ".npy") if probability_path else None

        if not os.path.exists(pred_path):
            click.echo(f"Error: Prediction file {gt_path} does not exist.")
            return

        gt_phases = read_phase_file(gt_path)
        pred_phases = read_phase_file(pred_path)

        if len(gt_phases) != len(pred_phases):
            click.echo(f"Error: Ground truth and prediction file have different lengths: {len(gt_phases)}/{len(pred_phases)}.")
            click.echo(f"Adjusting the length of the prediction file to match the ground truth file...")
            pred_phases = pred_phases[:len(gt_phases)]

        gt_phases = [phase_map[action] for action in gt_phases] # transform actions names to integers
        pred_phases = [phase_map[action] for action in pred_phases]
        
        accuracy = accuracy_score(gt_phases, pred_phases)
        f1 = f1_score(gt_phases, pred_phases, average='macro')
        edit = edit_score(gt_phases, pred_phases)
        
        

    

if __name__ == '__main__':
    evaluate()