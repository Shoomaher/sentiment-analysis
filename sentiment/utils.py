import json
from datetime import datetime
from os import listdir
from os.path import isdir, join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import multilabel_confusion_matrix
from tabulate import tabulate
from tensorflow.data import Dataset
from tensorflow.keras.metrics import Precision, Recall
from tensorflow_addons.metrics import F1Score

EMOJI_MAP = {
    'admiration': '👏',
    'amusement': '😂',
    'anger': '😡',
    'annoyance': '😒',
    'approval': '👍',
    'caring': '🤗',
    'confusion': '😕',
    'curiosity': '🤔',
    'desire': '😍',
    'disappointment': '😞',
    'disapproval': '👎',
    'disgust': '🤮',
    'embarrassment': '😳',
    'excitement': '🤩',
    'fear': '😨',
    'gratitude': '🙏',
    'grief': '😢',
    'joy': '😃',
    'love': '❤️',
    'nervousness': '😬',
    'optimism': '🤞',
    'pride': '😌',
    'realization': '💡',
    'relief': '😅',
    'remorse': '😔',
    'sadness': '😞',
    'surprise': '😲',
    'neutral': '⚪',
}


def load_classes(ds_dir):
    '''Load emotion classes from the dataset dir

    Args:
        ds_dir (str): path to dataset dir

    Returns:
        list(str): list of emotions
    '''
    with open(join(ds_dir, 'emotions.txt')) as classfile:
        return [line.rstrip('\n') for line in classfile]


def load_sentiments(ds_dir):
    '''Load sentiments and their classes from the dataset dir

    Args:
        ds_dir (str): path to dataset dir

    Returns:
        dict: sentiments: list of classes
    '''
    with open(join(ds_dir, 'sentiments.json')) as file:
        return json.load(file)


def load_dfs(ds_dir):
    '''Load dataframes from dataset dir

    Args:
        ds_dir (str): path to dataset dir

    Returns:
        pd.DataFrame: all dataframes unioned into a single frame
    '''
    parts = []
    for partname in listdir(ds_dir):
        if not partname.endswith('.tsv'):
            continue
        path = join(ds_dir, partname)
        part = pd.read_csv(path, sep='\t', names=['text', 'labels', 'id'])
        part.drop('id', axis=1, inplace=True)
        parts.append(part)
    full_df = pd.concat(parts, axis=0, ignore_index=True)
    assert full_df.index.is_unique
    return full_df


def oversample(df, threshold):
    '''Make oversampling of classes in dataframe that have qty of
    elements lower than the threshold value. Oversampling is performed to
    reach threshold. For example, if threshold is 500 and class 1 has qty
    of 270 elements, values will be duplicated to reach 500

    Args:
        df (pd.DataFrame): dataframe to oversample
        low_threshold (int): qty of elements under which the class is
            considered to be low

    Returns:
        pd.DataFrame: updated dataframe
    '''
    df = df.copy()
    single_labeled = ~df['labels'].str.contains(',')
    class_counts = df.loc[single_labeled, 'labels'].value_counts()
    low_classes = class_counts[class_counts < threshold].index.tolist()
    recombined = []
    for label in low_classes:
        class_mask = (df['labels'].str.contains(f'(^|,){label}(,|$)',
                                                regex=True))
        class_df = df[class_mask]
        df.drop(class_df.index, axis=0, inplace=True)
        class_df = class_df.sample(n=threshold, replace=True)
        recombined.append(class_df)
    return pd.concat([df] + recombined, axis=0, ignore_index=True)


def train_test_split(full_df,
                     fraction,
                     split_by_class=False,
                     test_only_singles=True,
                     random=None):
    '''Perform split on train/val/test

    Args:
        full_df (pd.DataFrame): full dataset to split
        fraction (float): percentage of split
        split_by_class (bool, optional): Get fraction class-wise instead
            of total split. Defaults to False.
        test_only_singles (bool, optional): Include only single-labelled
            into test. Defaults to False.
        random (int, optional): Random seed to use. Defaults to None.

    Raises:
        RuntimeError: if not all classes were included into test

    Returns:
        tuple: (train, val, test) dataframes
    '''
    if random is None:
        random = int(datetime.now().timestamp())
    print(f'Random seed: {random}')
    if split_by_class:
        train_parts = []
        val_parts = []
        test_parts = []
        for label, class_df in full_df.groupby('labels'):
            train_smpl = class_df.sample(frac=fraction, random_state=random)
            if test_only_singles and ',' in label:
                val_smpl = class_df.drop(train_smpl.index, axis=0)
                val_parts.append(val_smpl)
            else:
                test_smpl = class_df.drop(train_smpl.index, axis=0)
                test_parts.append(test_smpl)
                val_part = train_smpl.sample(frac=(1 - fraction),
                                             random_state=random)
                val_parts.append(val_part)
                train_smpl.drop(val_part.index, axis=0, inplace=True)
            train_parts.append(train_smpl)
        train_df = pd.concat(train_parts, axis=0, ignore_index=True)
        val_df = pd.concat(val_parts, axis=0, ignore_index=True)
        test_df = pd.concat(test_parts, axis=0, ignore_index=True)
    else:
        if test_only_singles:
            single_labeled = ~full_df['labels'].str.contains(',')
            all_labels = full_df[single_labeled]['labels'].unique()
            test_df = full_df[single_labeled].sample(frac=(1 - fraction),
                                                     random_state=random)
            test_labels = test_df['labels'].unique()
            if np.setdiff1d(all_labels, test_labels) or np.setdiff1d(
                    test_labels, all_labels):
                raise RuntimeError(
                    f'Test df has wrong set of labels: {test_labels}. '
                    f'Should be: {all_labels}')
        else:
            test_df = full_df.sample(frac=(1 - fraction), random_state=random)

        train_df = full_df.drop(test_df.index, axis=0)
        val_df = train_df.sample(frac=(1 - fraction), random_state=random)
        train_df.drop(val_df.index, axis=0, inplace=True)
    return (train_df, val_df, test_df)


def drop_labels(df, labels):
    '''Remove label from the dataset. It will be removed from the list of
    element labels. Elements without labels will be dropped. Please note
    that this function updates the original dataframe instead of
    returning updated.

    Args:
        df (pd.DataFrame): dataframe to remove label from
        label (list): list of labels to drop
    '''
    to_drop = [str(label) for label in labels]
    df['labels'] = df['labels'].str.split(',', expand=False)
    df['labels'] = df['labels'].map(
        lambda row: ','.join([x for x in row if x not in to_drop]))
    unlabelled = (df['labels'] == '')
    df.drop(df[unlabelled].index, axis=0, inplace=True)


def make_dataframes(ds_dir,
                    fraction,
                    split_by_class=False,
                    random=None,
                    test_only_singles=False,
                    drop_neutral=False,
                    oversample_low=False,
                    low_threshold=200):
    '''Create dataframes for modelling from the dataset dir. Read parts,
    combine, preprocess and split on train, val, test

    Args:
        ds_dir (str): path to dataset dir
        fraction (float): Percentage of split
        split_by_class (bool, optional): Apply fraction for train,
            validation, test split class-wise. Defaults to False.
        random (int, optional): Random seed to use. Defaults to None.
        test_only_singles (bool, optional): Include only single-labelled
            into test. Defaults to False.
        drop_neutral (bool, optional): Drop neutral labels. Defaults to
            False.
        oversample_low (bool, optional): Perform oversampling of low
            classes. Defaults to False.
        low_threshold (int, optional): Threshold of elements in class to
            consider it as low. Defaults to 200.

    Returns:
        tuple: (train, val, test) dataframes
    '''
    assert isdir(ds_dir)
    assert fraction > 0 and fraction < 1

    full_df = load_dfs(ds_dir)
    if drop_neutral:
        drop_labels(full_df, [27])
    train_df, val_df, test_df = train_test_split(full_df, fraction,
                                                 split_by_class,
                                                 test_only_singles)
    if oversample_low:
        train_df = oversample(train_df, low_threshold)
    for df in (train_df, val_df, test_df):
        assert df.index.is_unique
    train_df = train_df.sample(frac=1, random_state=random)
    train_df.reset_index(drop=True, inplace=True)
    return (train_df, val_df, test_df)


def make_ts_ds(df, classes, batch_size, prefetch_buffer):
    '''Convert dataframe to tensorflow dataset, with multi-hot encoded
    labels

    Args:
        df (pd.DataFrame): df to convert
        classes (list): list of classes
        batch_size (int): qty of elements for batching
        prefetch_buffer (tf.int64): buffersize for prefetching dataset

    Returns:
        tf.data.Dataset: tensorflow dataset for modelling
    '''
    classes_qty = len(classes)

    def encode_multihot(labels):
        encoded = np.zeros(classes_qty)
        for label in labels.split(','):
            encoded[int(label)] = 1
        return encoded

    labels_enc = df['labels'].apply(encode_multihot).tolist()
    ds = Dataset.from_tensor_slices((df['text'].tolist(), labels_enc))
    if batch_size:
        ds = ds.batch(batch_size)
    if prefetch_buffer:
        ds = ds.prefetch(prefetch_buffer)
    return ds


def create_metrics(classes, threshold):
    '''Create modelling metrics. Includes precision and recall for each
    class and for all classes. Also includes micro, macro and weighted
    F1-Score for all classes

    Args:
        classes (list): list of emotions to evaluate
        threshold (float): threshold value for class to activate

    Returns:
        list: list of metrics
    '''
    metrics = []
    for i, label in enumerate(classes):
        metrics.append(
            Precision(thresholds=[threshold],
                      class_id=i,
                      name=f'precision@{threshold}/{label}'))
        metrics.append(
            Recall(thresholds=[threshold],
                   class_id=i,
                   name=f'recall@{threshold}/{label}'))
    metrics.append(
        Precision(thresholds=[threshold], name=f'precision@{threshold}/all'))
    metrics.append(
        Recall(thresholds=[threshold], name=f'recall@{threshold}/all'))
    metrics.append(
        F1Score(threshold=threshold,
                num_classes=len(classes),
                name=f'f1_score@{threshold}/all',
                average='weighted'))
    metrics.append(
        F1Score(threshold=threshold,
                num_classes=len(classes),
                name=f'f1_score_micro@{threshold}/all',
                average='micro'))
    metrics.append(
        F1Score(threshold=threshold,
                num_classes=len(classes),
                name=f'f1_score_macro@{threshold}/all',
                average='macro'))
    return metrics


def print_metrics(metrics, floatfmt='.5f'):
    '''Pretty print metrics in table applying float format. Prin warning
    if metric is 0.

    Args:
        metrics (dict): metrics and their values
        floatfmt (str, optional): Format for floats. Defaults to '.5f'.
    '''
    stats_table = []
    for metric, value in metrics.items():
        if not value:
            print(f'WARNING!: {metric} = {value}')
        stats_table.append((metric, value))
    print(tabulate(stats_table, headers=['metric', 'value'],
                   floatfmt=floatfmt))


def plot_history(metrics):
    '''Plot metrics history over model fit epochs

    Args:
        metrics (dict): Metrics to plot
    '''
    fig = plt.figure(figsize=(10, 6))
    fig.tight_layout()
    plot_id = 1
    for metric, values in metrics.items():
        epochs = range(1, len(values) + 1)
        plt.subplot(2, 1, plot_id)
        plt.plot(epochs, values, label=metric)
        plt.title('Model fit history')
        plt.ylabel(metric)
        plt.xlabel('Epochs')
        plt.legend()


def test_examples(model, classes):
    '''Apply model for predicting default examples. Apply 0.5 as
    prediction threshold, warn if no such classes (model is not sure in
    its predictions) and take class with max prediction.

    Args:
        model (tf.keras.Model): model to apply
        classes (list): list of emotion classes
    '''
    examples = [
        'I am feeling great today!',
        'The weather is so good',
        'I have performed well at the university',
        'The war has started',
        'He is desperate in this cruel world',
        'I love the feeling when my girlfriend hugs me',
        'I hate monday mornings',
        'Merry Christmas! I told Santa you were good this year and '
        'asked him to bring you a year full of joy and pleasure ',
        'brilliant! Such a detailed review, it was a pleasure, thank you! '
        'Guys, make sure you find time to read :) Aaaaand you can actually choose sth new)',
        'I have the new pan for pancakes.',
        'Relax, bro. Take it easy',
        "WTF? Are they kidding us? I'm gonna argue with the manager!",
        'OMG, yep!!! That is the final answer! Thank you so much!',
        'I am so glad this is over',
        'Sorry, I feel bad for having said that',
        'Happy birthday, my friend! I wish you a lot of success!',
        'What a shame! I will never talk to him',
        "What if she knows? We don't know what to do",
        'WOW! I am really into cinema',
        "What if I don't pass the exam? I will never get this driving license!",
        'I miss my grandad. I am feeling so lonely after her death. '
        'I just lost my closest person..',
        'Skipping lessons is so miserable for Oxford stundets!',
        'Have a rest, my boy. You had a long trip. I will make us tea and bring a cake.',
        'What a man! My son got the highest rate and will study in Cambridge. '
        'Our family is so proud of him!',
        "Mmmm, delicious. That's totally the best pasta in Italy!",
    ]
    model_scores = model(tf.constant(examples))
    np_classes = np.array(classes)
    for idx, predictions in enumerate(model_scores):
        predicted = (predictions >= 0.5).numpy()
        if predicted.any():
            emotions = np_classes[predicted]
        else:
            emotions = []
        with_emojis = []
        for emotion in emotions:
            try:
                emoji = EMOJI_MAP[emotion]
            except KeyError:
                emoji = 'N/A'
            with_emojis.append(f'{emotion} {emoji}')
        print('{}: {}'.format(examples[idx], ' '.join(with_emojis)))


def plot_conf_mtrx_all(model, test_ds, classes, normalized=True):
    '''Plot confusion matrix for all classes using all vs all strategy.
    Be aware that only single-labelled classes are needed. Emotion with
    max prediction from model will be selected

    Args:
        model (tf.keras.Model): model to evaluate
        test_ds (tf.data.Dataset): dataset for test
        classes (list): list of emotions
        normalized (bool, optional): Normalize matrix. Defaults to True.
    '''
    num_classes = len(classes)
    mtrx = tf.zeros([num_classes, num_classes], dtype=tf.int32)
    for texts, labels in test_ds:
        predictions = model(texts)
        mtrx += tf.math.confusion_matrix(tf.argmax(labels, axis=1),
                                         tf.argmax(predictions, axis=1),
                                         num_classes=num_classes)
    mtrx = mtrx.numpy()
    if normalized:
        mtrx = mtrx.astype('float') / mtrx.sum(axis=1)[:, np.newaxis]
        mtrx[np.isnan(mtrx)] = 0

    fig, ax = plt.subplots(figsize=(25, 25))
    im = ax.imshow(mtrx, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(mtrx.shape[1]),
           yticks=np.arange(mtrx.shape[0]),
           xticklabels=classes,
           yticklabels=classes,
           xlabel='Predicted label',
           ylabel='True label')
    ax.set_xticklabels(classes, rotation=45)
    thresh = mtrx.max() / 2.
    for i in range(mtrx.shape[0]):
        for j in range(mtrx.shape[1]):
            ax.text(j,
                    i,
                    format(mtrx[i, j], '.4f'),
                    ha="center",
                    va="center",
                    color="white" if mtrx[i, j] > thresh else "black")
    fig.tight_layout()
    plt.show()


def plot_conf_mtrx_per_class(model,
                             test_ds,
                             classes,
                             threshold=0.5,
                             select_max_class=False):
    '''Plot confusion matrixes for each class separately

    Args:
        model (tf.keras.Model): model to test
        test_ds (tf.data.Data): dataset for test
        classes (list): list of emotions
        threshold (float): threshold prediction value above which the
            class is activated. Defaults to 0.5
        select_max_class (bool, optional): Select class with max
            prediction. If not - use threshold. Defaults to False.
    '''
    cm_sum = None
    for x, y_true in test_ds:
        y_pred = model.predict(x)
        if select_max_class:
            max_indices = np.argmax(y_pred, axis=1)
            y_top = np.zeros_like(y_pred)
            y_top[np.arange(y_pred.shape[0]), max_indices] = 1
            y_pred = y_top
        else:
            y_pred = (y_pred >= threshold)
        cm_batch = multilabel_confusion_matrix(y_true, y_pred)
        if cm_sum is None:
            cm_sum = cm_batch
        else:
            cm_sum += cm_batch

    fig, axs = plt.subplots(10,
                            3,
                            figsize=(15, 50),
                            sharex=False,
                            sharey=False,
                            gridspec_kw={
                                'hspace': 0.2,
                                'wspace': 0.3
                            })
    for i, label in enumerate(classes):
        row_idx = i // 3
        col_idx = i % 3
        axis = axs[row_idx, col_idx]
        im = axis.imshow(cm_sum[i], interpolation='nearest', cmap=plt.cm.Blues)
        cbar = axis.figure.colorbar(im, ax=axis)
        cbar.ax.set_ylabel('Count', rotation=-90, va="bottom")
        for row in range(2):
            for col in range(2):
                axs[row_idx, col_idx].text(col,
                                           row,
                                           str(cm_sum[i][row][col]),
                                           ha='center',
                                           va='center',
                                           color='black',
                                           fontsize=8)
        axis.set_xticks(np.arange(2))
        axis.set_yticks(np.arange(2))
        axis.set_xticklabels(['False', 'True'], fontsize=8)
        axis.set_yticklabels(['False', 'True'], fontsize=8)
        axis.set_xlabel('Predicted label', fontsize=8)
        axis.set_ylabel('True label', fontsize=8)
        axis.set_title(f'Confusion matrix for class {label}', fontsize=8)
    plt.show()


def get_class_counts(df):
    '''Get class counts in dataframe

    Args:
        df (pd.DataFrame): split not encoded labels and create class
            counts

    Returns:
        np.ndarray: 1d array of counts of each class
    '''
    all_labels = df['labels'].str.split(',', expand=True).values.flatten()
    all_labels = all_labels.astype(np.float16)
    all_labels = all_labels[~np.isnan(all_labels)]
    _, class_counts = np.unique(all_labels, return_counts=True)
    return class_counts


def plot_class_distr(df, classes, title):
    '''Plot classes distrubution in dataframe

    Args:
        df (pd.DataFrame): dataframe to plot classes
        classes (list): list of emotions
        title (str): title for plot
    '''
    counts = get_class_counts(df)
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.bar(classes, counts, width=0.5)
    ax.set_title(title)
    ax.set_xlabel('Classes')
    ax.set_ylabel('Elements')
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.set_xticks(np.arange(len(classes)))
    for idx, qty in enumerate(counts):
        ax.text(idx, qty + 15, str(qty), ha='center', fontweight='bold')
    plt.show()


def map_sentiments(all_classes, all_sentiments):
    '''Create map of labels (0..max class) to sentiments. Neutral is
    separated

    Args:
        all_classes (list): list of all emotions
        all_sentiments (dict): dict of sentiments and their emotions

    Returns:
        dict: map of emotion labels to sentiment labels
    '''
    mapped = {}
    for label, (sentiment, classes) in enumerate(all_sentiments.items()):
        for class_ in classes:
            mapped[all_classes.index(class_)] = label
    mapped[27] = label + 1  # neutral
    return mapped


def to_sentiments(batch, sentiment_map):
    '''Convert batch of multi-hot encoded emotion labels to multi-hot
    encoded sentiments labels

    Args:
        batch (np.ndarray): batch of labels (labels are rows)
        sentiment_map (dict): map of emotion class nums to sentiment
            class nums

    Returns:
        np.ndarray: batch of sentiments encoded
    '''
    sentimented = np.zeros((batch.shape[0], max(sentiment_map.values()) + 1))
    for el_id in range(batch.shape[0]):
        for label in range(batch.shape[1]):
            if batch[el_id, label] > 0:
                sentiment = sentiment_map[label]
                sentimented[el_id, sentiment] += 1
    return sentimented


def score_test(test_ds, model, metrics, threshold=0.5, sentiment_map=None):
    '''Score test dataframe and run metrics

    Args:
        test_ds (tf.data.Dataset): dataset for test
        model (tf.keras.Model): model to evaluate
        metrics (list): list of tensorflow metrics
        threshold (float): threshold value over which predicted class is
            activated. Defautls to 0.5
        sentiment_map (dict, optional): Map of sentiments to class. If
            provided - check sentiments prediction. Defaults to None.

    '''
    for texts, true in test_ds:
        predictions = model(texts)
        predicted = (predictions.numpy() >= threshold).astype(int)
        if sentiment_map:
            predicted = to_sentiments(predicted, sentiment_map)
            true = to_sentiments(true, sentiment_map)
        for m in metrics:
            m.update_state(true, predicted)
