import pandas as pd
import numpy as np
import cv2
import os
import sklearn
from os.path import join
import matplotlib.pyplot as plt
from matplotlib import colormaps
from sklearn import metrics


class NameMapping:
    def __init__(self, mapping_path, old_name='mapped_video_id', new_name='new_video_id') -> None:
        """
        Initialize the NameMapping class.
        Args:
            mapping_path (str): The path to the mapping file.
            old_name (str, optional): The column name for the old video ID. Defaults to 'mapped_video_id'.
            new_name (str, optional): The column name for the new video ID. Defaults to 'new_video_id'.
        """
        self.mapping = pd.read_csv(mapping_path)
        self.old_name = old_name
        self.name = new_name
    
    def get_new_name(self, old_name):
        """
        Get the new video ID based on the old video ID.
        Args:
            old_name (str): The old video ID.
        Returns:
            str: The new video ID corresponding to the old video ID. Returns None if not found.
        """
        value = self.mapping.loc[self.mapping[self.old_name] == old_name, self.name]
        if len(value) == 0:
            return None
        return value.item()
    
    def get_old_name(self, new_name):
        """
        Get the old video ID based on the new video ID.
        Args:
            new_name (str): The new video ID.
        Returns:
            str: The old video ID corresponding to the new video ID. Returns None if not found.
        """
        value = self.mapping.loc[self.mapping[self.name] == new_name, self.old_name]
        if len(value) == 0:
            return None
        return value.item()
    
    def get_gt_from_id(self, orig_id):
        value = self.mapping.loc[self.mapping['Video ID'] == orig_id, self.name]
        if len(value) == 0:
            return None
        return os.path.splitext(value.item())[0] + '.txt'
    
    def get_name(self, query, name, identifier):
        """
        Retrieves the name associated with the given identifier from the mapping .
        Args:
            query (str): The column name to query in the mapping DataFrame.
            name (str): The column name to retrieve the new identifier from.
            identifier: The identifier to search for in the mapping.
        Returns:
            str or None: The name associated with the identifier, or None if no match is found.
        """
        value = self.mapping.loc[self.mapping[query] == identifier, name]
        if len(value) == 0:
            return None
        return os.path.splitext(value.item())[0]


def get_frame_from_time(time, fps=30):
    """
    Converts a time in the format 'MM:SS' to the corresponding frame number based on the given frames per second (fps).
    Args:
        time (str): The time in the format 'MM:SS'.
        fps (int, optional): The frames per second. Defaults to 29.
    Returns:
        int or None: The corresponding frame number if the time is valid, None otherwise.
    """
    try: 
        min, sec = time.split(':')[:2]
        time_secs = int(min) * 60 + int(sec)
        frame = time_secs * fps
        return int(np.ceil(frame))
    except:
        return None


def get_video_info(file):
    """
    Get the duration, fps and frame count of a video file.
    Args:
        file (str): The path to the video file.
    Returns:
        tuple: A tuple containing the duration (in seconds) and frame count and the fps (in frames/seconds)
    """
    cap = cv2.VideoCapture(file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count/fps
    return duration, frame_count, fps


def process_annotation(row, video_dir, cmapping, col_name='mapped_video_id', fps=30, start_idx=7):
    """
    Process the annotation for a given row of data.
    Args:
        row (pandas.Series): The row of data containing the annotation information.
        video_dir (str): The directory where the video files are located.
        cmapping (dict): A dictionary mapping class names to class integer labels.
        col_name (str, optional): The name of the column containing the video file names. Defaults to 'mapped_video_id'.
        fps (int, optional): The frames per second of the video. Defaults to 30.
        start_idx (int, optional): The starting index of the annotation information in the row. Defaults to 7.
    Returns:
        numpy.ndarray: The processed annotation as a numpy array.
    """
    duration, vid_len, orig_fps = get_video_info(join(video_dir, row[col_name]))
    #print(f"Working on video {row.index[0]} with {int(vid_len)} frames and {fps} fps...")
    new_vid_len = int(vid_len / orig_fps * fps)

    annotation = np.zeros(new_vid_len)
    for i in range(start_idx, start_idx + 40, 2):
        start_frame = get_frame_from_time(row.iloc[i], fps=fps)
        end_frame = get_frame_from_time(row.iloc[i+1], fps=fps)
        if start_frame == None or end_frame == None:
            continue
        
        cur_class = "_".join(row.index[i].split('_')[:-1])
        annotation[start_frame:end_frame] = int(cmapping[cur_class])
    
    return annotation


def parse_remark_cell(cell1, cell2, cmapping, fps=30):
    """
    Parses the remark cell with the format <classname>_start/stop: <timestep> and extracts the class, start time, and end time.
    Time should be in the format MM:SS.
    Args:
        cell1 (str): The first cell of the remark.
        cell2 (str): The second cell of the remark.
        fps (int, optional): Frames per second. Defaults to 30.
    Returns:
        tuple: A tuple containing the start time (int), end time (int), and class (str).
    """
    cell_class = cell1.split(':')[0]
    cell_class = "_".join(cell_class.split('_')[:-1])
    
    if cmapping.get(cell_class) is None:
        print(f'Class {cell_class} not found in mapping')
    
    start_time = ":".join(cell1.split(':')[1:])
    start_time = get_frame_from_time(start_time, fps=fps)
    
    end_time = ":".join(cell2.split(':')[1:])
    end_time = get_frame_from_time(end_time, fps=fps)
    
    return start_time, end_time, cell_class


def f_score(recognized, ground_truth, overlap, bg_class=["background"]):
    p_label, p_start, p_end = get_labels_start_end_time(recognized, bg_class)
    y_label, y_start, y_end = get_labels_start_end_time(ground_truth, bg_class)

    tp = 0
    fp = 0

    hits = np.zeros(len(y_label))

    for j in range(len(p_label)):
        intersection = np.minimum(p_end[j], y_end) - np.maximum(p_start[j], y_start)
        union = np.maximum(p_end[j], y_end) - np.minimum(p_start[j], y_start)
        IoU = (1.0*intersection / union)*([p_label[j] == y_label[x] for x in range(len(y_label))])
        # Get the best scoring segment
        idx = np.array(IoU).argmax()

        if IoU[idx] >= overlap and not hits[idx]:
            tp += 1
            hits[idx] = 1
        else:
            fp += 1
    fn = len(y_label) - sum(hits) 
    return float(tp), float(fp), float(fn)


def acc_phase(recognized, ground_truth, overlap=0.25, bg_class=[""]):
    tp, fp, fn = f_score(recognized, ground_truth, overlap, bg_class)
    total_instances = tp + fp + fn
    accuracy = tp /  total_instances if total_instances > 0 else 0
    #print(tp, fp, fn, total_instances)
    return accuracy * 100.0


def write_str_to_file(filename, content):
    with open(filename, 'w') as f:
        f.write(content)

        
def load_action_map(mapping_file):
    actions = read_file(mapping_file).split('\n')[:-1]
    actions_dict = dict()
    for a in actions:
        actions_dict[a.split()[1]] = int(a.split()[0])
    
    return actions_dict

        
def read_file(path):
    with open(path, 'r') as f:
        content = f.read()
        f.close()
    return content


def get_labels_start_end_time(frame_wise_labels, bg_class=[""]):
    labels = []
    starts = []
    ends = []
    last_label = frame_wise_labels[0]
    if frame_wise_labels[0] not in bg_class:
        labels.append(frame_wise_labels[0])
        starts.append(0)
    for i in range(len(frame_wise_labels)):
        if frame_wise_labels[i] != last_label:
            if frame_wise_labels[i] not in bg_class:
                labels.append(frame_wise_labels[i])
                starts.append(i)
            if last_label not in bg_class:
                ends.append(i)
            last_label = frame_wise_labels[i]
    if last_label not in bg_class:
        ends.append(i)
    return labels, starts, ends


def survey(results, category_names, colors, ax, fig_size=(25, 3)):
    data = np.array(list(results.values()))
    data_cum = data.cumsum(axis=1)
    
    ax.invert_yaxis()
    ax.xaxis.set_visible(False)
    ax.set_xlim(0, np.sum(data))
    
    for i, (colname, color) in enumerate(zip(category_names, colors)):
        widths = data[:, i]
        starts = data_cum[:, i] - widths
        rects = ax.barh([""], widths, left=starts, height=0.75, label=colname, color=color)

        r, g, b = color
        text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
        ax.bar_label(rects, labels=[colname], label_type='center', color=text_color, rotation=90 if fig_size[1] >=3 else 0 )
    return ax


def get_labels(dataset_name):
    if "cataract101" in dataset_name:
        labels = ['action_start', 'incision', 'viscous_agent_injection', 'rhexis', 'hydrodissection', 'phacoemulsificiation', 'irrigation_aspiration', 
                  'capsule_polishing', 'lens_implant_settingup', 'viscous_agent_removal', 'tonifying_antibiotics', 'action_end']
    elif "ieee50" in dataset_name:
        labels = ['background', 'toric_marking', 'implant_ejection', 'incision', 'viscodilatation', 'capsulorhexis', 'hydrodissetion', 
                  'nucleus_breaking', 'phacoemulsification', 'vitrectomy', 'irrigation_aspiration', 'preparing_implant', 'manual_aspiration', 
                  'implantation', 'positioning', 'ovd_aspiration', 'suturing', 'sealing_control', 'wound_hydratation']
    elif "paxos" in dataset_name:
        labels = ['background', 'Fundus_Visibile']
    else:
        labels = ['background', 'main_incision_entry', 'cautery', 'peritomy', 'tunnel_suture', 'hydroprocedure', 'conjunctival_cautery', 'tunnel', 
                  'nucleus_prolapse', 'OVD_IOL_insertion', 'sideport', 'scleral_groove', 'OVD_injection', 'cortical_wash', 'OVD_wash', 'stromal_hydration', 
                  'nucleus_delivery', 'incision', 'capsulorrhexis', 'AB_injection_and_wash', 'SR_bridle_suture']
    return labels

######## VISUALIZATION ########

def get_colormap(labels, cm_name='viridis'):
    colormap = colormaps[cm_name]
    colors = colormap(np.linspace(0, 1, len(labels)))
    color_mapping = {labels[i]: colors[i, :][:3] for i in range(len(labels))}
    return color_mapping


def plot_graphs_for_dataset(dataset_name, split, output_dir):
    if not os.path.exists(output_dir): os.mkdir(output_dir)
    files = os.listdir(f"./results/{dataset_name}/split_{split}/")
    files.sort()
    cases = files[::2]    
    
    labels = get_labels(dataset_name)
    color_mapping = get_colormap(labels)
    
    print('Plotting results: ', end='')
    for case in cases:
        print(f'{case}', end=', ' if case != cases[-1] else '\n')
        pred_path = f"./results/{dataset_name}/split_{split}/{case}"
        gt_path = f"./data/{dataset_name}/groundTruth/{case}.txt"

        predictions = read_file(pred_path).split('\n')[1].split()
        if predictions[-1] == '': predictions = predictions[:-1]
        ground_truth = read_file(gt_path).split('\n')
        if ground_truth[-1] == '': ground_truth = ground_truth[:-1]
        
        labels_gt, start_gt, end_gt = get_labels_start_end_time(ground_truth)
        labels_pred, start_pred, end_pred = get_labels_start_end_time(predictions)

        data_gt = {"Ground Truth": [end_gt[i] - start_gt[i] for i, _ in enumerate(start_gt)]}
        data_pred = {"Prediction": [end_pred[i] - start_pred[i] for i, _ in enumerate(start_pred)]}
        
        fig, axs = plt.subplots(2, 1, figsize=(25, 6), sharex=True, sharey=True, layout="tight")
        axs[0].axis('off')
        axs[1].axis('off')

        colors_gt = [color_mapping[elm] for elm in labels_gt]
        ax = survey(data_gt, labels_gt, colors_gt, axs[0])
        colors_pred = [color_mapping[elm] for elm in labels_pred]
        ax2 = survey(data_pred, labels_pred, colors_pred, axs[1])
        
        plt.show()
        fig.savefig(output_dir + case + ".png")
        plt.close()

        
def plot_confusion_matrix(ground_truth, predictions, actions_dict, output_dir, normalized='true'):
    action_labels = list(actions_dict.keys())
    figsize = 40 if 'sics' in output_dir else 20
    mat = metrics.confusion_matrix(ground_truth, predictions, labels=action_labels)
    fig, ax = plt.subplots(figsize=(figsize,figsize))
    suffix = 'Normalized' if normalized is not None else 'Default'
    form = ".2%" if normalized is not None else "d"
    print(f"Plotting confusion matrix, plot-mode: {suffix.lower()}...")
    plt.rcParams.update({'font.size': 20})

    display = metrics.ConfusionMatrixDisplay.from_predictions(ground_truth, predictions, labels=action_labels, 
                                                              normalize=normalized, display_labels=action_labels)
    display.plot(ax=ax, cmap="Blues", values_format=form, colorbar=False)
    # ax.set_title(f"Confusion Matrix - {suffix}")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, fontsize=24)
    ax.set_yticklabels(['\n_'.join(l.get_text().split('_')) for l in ax.get_yticklabels()], rotation=0)
    ax.tick_params(labelsize=24)
    plt.rc("axes", titlesize=24) 
    plt.rc("axes", labelsize=24)
    plt.rc("font", size=24)

    plt.tight_layout()
    plt.show()
    fig.savefig(os.path.join(output_dir, f"confusion_matrix_{suffix.lower()}.png"))
    plt.close()


######## METRICS #############

def levenstein(p, y, norm=False):
    m_row = len(p)
    n_col = len(y)
    D = np.zeros([m_row+1, n_col+1], np.float64)
    for i in range(m_row+1):
        D[i, 0] = i
    for i in range(n_col+1):
        D[0, i] = i

    for j in range(1, n_col+1):
        for i in range(1, m_row+1):
            if y[j-1] == p[i-1]:
                D[i, j] = D[i-1, j-1]
            else:
                D[i, j] = min(D[i-1, j] + 1,
                              D[i, j-1] + 1,
                              D[i-1, j-1] + 1)

    if norm:
        score = (1 - D[-1, -1]/max(m_row, n_col)) * 100
    else:
        score = D[-1, -1]

    return score


def edit_score(recognized, ground_truth, norm=True, bg_class=[""]):
    P, _, _ = get_labels_start_end_time(recognized, bg_class)
    Y, _, _ = get_labels_start_end_time(ground_truth, bg_class)
    return levenstein(P, Y, norm)


def collaps_confusion_matrix(y_true, y_pred):
    cm = metrics.confusion_matrix(y_true, y_pred)
    tp, tn, fp, fn = 0, 0, 0, 0
    for row_idx in range(cm.shape[0]):
        tp += cm[row_idx, row_idx]
        fp += cm[row_idx, :].sum() - cm[row_idx, row_idx]
        fn += cm[:, row_idx].sum() - cm[row_idx, row_idx]
        tn += cm.sum() - tp - fp - fn 
    return tp, tn, fp, fn


def prepare_results(ground_truth_dir, results_dir, action_dict, sample_rate=1):
    results_files = os.listdir(results_dir)
    probability_files = [file for file in results_files if '.npy' in file]

    actions2num = np.vectorize(lambda d: action_dict[d])
    results = {"labels": [], "predictions": [], "probs": []}

    for probs_files in probability_files:
        probs = np.load(join(results_dir, probs_files))[:, 0::sample_rate]
        prediction = open(join(results_dir, probs_files.replace('.npy', '')), 'r').read().split('\n')[1].split()
        ground_truth = open(join(ground_truth_dir, probs_files.replace('.npy', '.txt')), 'r').read().split('\n')
        ground_truth = ground_truth[0::sample_rate]
        ground_truth = ground_truth[:probs.shape[1]]
        prediction = prediction[0::sample_rate]
        prediction = prediction[:probs.shape[1]]
        assert probs.shape[1] == len(ground_truth)

        ground_truth = actions2num(ground_truth)
        prediction = actions2num(prediction)

        results["labels"].append(ground_truth)
        results["predictions"].append(prediction)
        results["probs"].append(probs)
    return results
# dataset = "sics73_rgb"
# plot_graphs_for_dataset(dataset, 0, f"./results/{dataset}/figures/")

