import os
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from os.path import join, isdir, basename
from sklearn import base
from sklearn.metrics import f1_score, accuracy_score, average_precision_score
from utils import read_phase_file, load_action_map, edit_score 


GROUND_TRUTH_DIR = 'groundTruth'
RESULTS_DIR = 'results'
MAPPING_PATH = 'mapping.txt'


def calc_metrics(gt_path, pred_path, phase_map):
    """ Calculate accuracy, f1 score, and edit score for a single pair of ground truth and prediction files. """

    if not os.path.exists(pred_path):
        print(f"Error: Prediction file {pred_path} does not exist.")
        return 0.0, 0.0, 0.0

    gt_phases = read_phase_file(gt_path)
    pred_phases = read_phase_file(pred_path)
        
    if len(gt_phases) != len(pred_phases):
        # print(f"Error: Ground truth and prediction file have different lengths: {len(gt_phases)}/{len(pred_phases)}.")
        # print(f"Adjusting the length of the prediction file to match the ground truth file...")
        pred_phases = pred_phases[:len(gt_phases)]

    gt_phases = [phase_map[action] for action in gt_phases] # transform actions names to integers
    pred_phases = [phase_map[action] for action in pred_phases]
    
    accuracy = accuracy_score(gt_phases, pred_phases)
    f1 = f1_score(gt_phases, pred_phases, average='macro')
    edit = edit_score(gt_phases, pred_phases)
    return accuracy, f1, edit


def accumulate_result_scores(results_path, results_dirs, gt_dir, gt_files):
    """ Accumulate results from all submissions in one dataframe. """
    phase_map = load_action_map(MAPPING_PATH) # load action to integer mapping
    gt_files = [join(gt_dir, f) for f in gt_files if f.endswith('.txt')]
    video_names = [os.path.splitext(f)[0] for f in os.listdir(gt_dir) if f.endswith('.txt')]

    columns = ['video_id']
    for res in results_dirs: # add columns for each result directory
        columns.extend([f"{res}_accuracy", f"{res}_f1", f"{res}_edit"])
    results = pd.DataFrame(columns=columns)

    for vid in video_names: # initialize dataframe with video ids
        results = pd.concat([results, pd.DataFrame({'video_id': [vid]})], ignore_index=True)

    for res_by_team in results_dirs:
        res_dir_path = join(results_path, res_by_team)
        print(f"Processing results in directory: {res_dir_path}")
        res_files = os.listdir(res_dir_path)
        res_files = [f for f in res_files if f.endswith('.txt')]

        for vid in video_names:
            gt_path = join(gt_dir, f"{vid}.txt")
            pred_path = join(res_dir_path, f"{vid}.txt")

            accuracy, f1, edit = calc_metrics(gt_path, pred_path, phase_map)

            results.loc[results['video_id'] == vid, f"{basename(res_by_team)}_accuracy"] = accuracy
            results.loc[results['video_id'] == vid, f"{basename(res_by_team)}_f1"] = f1
            results.loc[results['video_id'] == vid, f"{basename(res_by_team)}_edit"] = edit

    return results


def determine_team_ranks(results, results_dirs, mode=np.median):
    """ Perform Wilcoxon signed-rank test for each pair of submissions and determine team ranks. """
    metrics = ['accuracy', 'f1', 'edit']
    alpha = 0.05
    team_ranks = {basename(d): {metric: 0 for metric in metrics} for d in results_dirs}

    for metric in metrics:
        for team in results_dirs:
            worse_than = 0
            team_name = basename(team)    
            for other_team in results_dirs:
                other_team_name = basename(other_team)
                if team_name is other_team_name:
                    continue
                # print(f"\nComparing {team_name} vs {other_team_name}:")
                
                team_score = results[f"{team_name}_{metric}"].to_list()
                other_team_score = results[f"{other_team_name}_{metric}"].to_list()
                stat, p = wilcoxon(x=team_score, y=other_team_score, alternative='less', zero_method='zsplit')
                # print(f"  {metric}: statistic={None}, p-value={p}")
                if p < alpha:
                    worse_than += 1
            print(f"{team_name} is significantly worse than {worse_than} out of {len(results_dirs)} teams in metric {metric}.")
            team_ranks[team_name][metric] = len(results_dirs) - worse_than

    team_ranks = {team: mode(list(scores.values())) for team, scores in team_ranks.items()}
    return team_ranks


def run():
    """ Main function to run the ranking evaluation. """
    print("Running point-based significance ranking evaluation...")

    # Prepare list of result directories and ground truth files
    results_dirs = os.listdir(RESULTS_DIR) 
    results_dirs = [d for d in results_dirs if isdir(join(RESULTS_DIR, d))]
    gt_files = os.listdir(GROUND_TRUTH_DIR)
    gt_files = [f for f in gt_files if f.endswith('.txt')]


    # STEP 1: Accumulate results from all submission in one dataframe
    results = accumulate_result_scores(RESULTS_DIR, results_dirs, GROUND_TRUTH_DIR, gt_files)
    if not os.path.exists('evaluation_results.xlsx'):
        results.to_excel('evaluation_results.xlsx', index=False)
        print("Evaluation results saved to evaluation_results.xlsx")

    # STEP 2: Perform Wilcoxon signed-rank test for each pair of submissions
    team_ranks = determine_team_ranks(results, results_dirs)
    
    sorted_teams = sorted(team_ranks.items(), key=lambda x: x[1], reverse=True)
    print("\nFinal Rankings:")
    for rank, (team, score) in enumerate(sorted_teams, start=1):
        print(f"Rank {rank}: {team} (Score: {score})")


if __name__ == '__main__':
    run()