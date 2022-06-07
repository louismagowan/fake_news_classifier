# IMPORTS
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


############ ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
######### PRE-PROCESSING


def load_data(filepath_fake, filepath_true):
    """"Function to quickly read in our data, merge it and
    convert dates to datetime. There were some messy dates that needed to
    be handled separately.
    ----------------------------------------------------------------------
    filepath_fake = str, path to csv containing fake news
    filepath_true = str, path to csv containing true news
    """
    # Read in data
    fake = pd.read_csv(filepath_fake)
    true = pd.read_csv(filepath_true)

    # Add column to indicate if fake (1) or true (0)
    fake["fake_news"] = 1
    true["fake_news"] = 0

    # Merge
    all_news = pd.concat([fake, true])
    # Split into data with correct dates
    clean = all_news[~all_news.date.str.contains("-|\\[", regex = True)].copy()
    # And incorrect dates
    messy = all_news[all_news.date.str.contains("-|\\[", regex = True)].copy()
    # Remove the urls accidentally stored as dates
    messy.loc[messy.date.str.contains("/|\\["), "date"] = np.NaN
    # Convert the other dates that were in the wrong format into datetime
    messy.date = pd.to_datetime(messy.date)
    # Also convert the cleaner dates
    clean.date = pd.to_datetime(clean.date)

    df = pd.concat([clean, messy]).reset_index(drop = True)

    return df

def prep_text(data,
            drop_dups_text = True,
            drop_dups_title = True,
            drop_dups_all = False):
    """
    Function to deduplicate (according to user-specified subset) and return
    df with title and text columns joined together.
    ----------------------------------------------------------------------
    drop_dups_text = bool, if True deduplicates on text column
    drop_dups_title = bool, if True deduplicates on title column
    drop_dups_all = bool, if True deduplicates on combination of text and title
    """
    # create copy to prevent in place changes
    df = data.copy()

    # Make a joined text column
    df["all_text"] = df.title.str.strip() + " " + df.text.str.strip()
    # Remove certain words associated with information leak
    df["all_text"] = df["all_text"]\
                        .str.replace("reuters|true|false|washington|verified|politifact|donald trump|21st century wire",
                         "", 
                         case = False, 
                         regex = True)
    # Rigorous deduplication as there were several duplicates found
    # Drop duplicated text
    if drop_dups_text:
        df = df.drop_duplicates(subset = "text").reset_index(drop = True)

    # Drop duplicated titles
    if drop_dups_title:
        df = df.drop_duplicates(subset = "title").reset_index(drop = True)

    # Drop if both text and title are the same
    # Added in to give extra data processing control
    if drop_dups_all:
        df = df.drop_duplicates(subset = "all_text").reset_index(drop = True)

    return df[["all_text", "fake_news"]]


def tokenize_padder(train_text, test_text,
                   chars_to_filter = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                    oov_token = "OOV",
                    maxlen = 500,
                    padding = "pre",
                    truncating = "post"
                   ):
    """
    Function to train a Keras Tokenizer on training text data, then use that to
    generate sequences for both training and text data. It then pre-pads those
    sequences that are too short and post-trims any that are too long, ensuring
    they are all the same length. Returns fitted Tokenizer object,
    padded training data and padded test data.
    Mean/median token counts in the dataset were around 400so a maxlen of 500
    may be sufficient.
    ----------------------------------------------------------------------
    chars_to_filter = str, regex pattern of chars to be removed by the
    tokenizer
    oov_token = str, string representation for out-of-vocabulary tokens
    max_len = int, maximum length of sequences
    padding = str, whether sequences should be padded at the front or back
    truncating = str, whether sequences should be truncated at the front or back
    """
    # Create tokenizer
    tokenizer = Tokenizer(filters = chars_to_filter,
                          oov_token = oov_token)

    # Fit tokenizer on training data only
    tokenizer.fit_on_texts(train_text)

    # Generate sequences
    train_sequences = tokenizer.texts_to_sequences(train_text)
    test_sequences = tokenizer.texts_to_sequences(test_text)

    # Pad and trim sequences
    # Pre-padding is empirically better for sequence modelling
    # Post-truncating ensures the titles are included in observations
    train_padded = pad_sequences(train_sequences, maxlen = maxlen, padding = padding, truncating = truncating)
    test_padded = pad_sequences(test_sequences, maxlen = maxlen, padding = padding, truncating = truncating)

    return tokenizer, train_padded, test_padded


# PLOTTING + FORMATTING
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def set_plot_config():
    """"Function to set-up Matplotlib plotting config
    for neater graphs"""
    plt.rcParams["figure.figsize"] = (17, 8)
    font = {'family' : 'serif',
            'weight' : 'normal',
            'size'   : 22}
    plt.rc('font', **font)


# Class to make print statements prettier
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def plot_loss(history, xlabel = "Epochs", ylabel = "Loss",
             title = "Loss vs Epochs", ylim_top = None,
             metric = "loss", metric2 = "val_loss",
             metric_label = "Training Loss", metric2_label = "Validation Loss"):
    """Function to plot model loss.
    Takes history, a model history instance.
    Displays plot inline"""
    # Show training and validation loss vs epochs
    fig, ax = plt.subplots()
    ax.plot(history.history[metric], label = metric_label, lw = 3)
    # Add in control if you only want to plot one metric
    if metric2:
        ax.plot(history.history[metric2], label = metric2_label, lw = 3)
        ax.legend()
    # Add titles
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(top = ylim_top)



def plot_acc(history, xlabel = "Epochs", ylabel = "Accuracy",
             title = "Accuracy vs Epochs", ylim_top = None,
             metric = "accuracy", metric2 = "val_accuracy",
             metric_label = "Training Accuracy", metric2_label = "Validation Accuracy"):
    """Function to plot model accuracy.
    Takes history, a model history instance.
    Displays plot inline"""
    # Show training and validation loss vs epochs
    fig, ax = plt.subplots()
    ax.plot(history.history[metric], label = metric_label, lw = 3, color = "lightgreen")
    # Add in control if you only want to plot one metric
    if metric2:
        ax.plot(history.history[metric2], label = metric2_label, lw = 3, color = "green")
        ax.legend()
    # Add titles
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(top = ylim_top)


############ ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Misc

def tf_message_toggle(log_level = "2"):
    """"Function to control the amount of messages that TF
    displays. e.g. Use it to prevent the setting GPU to xyz core
    info messages that TensorFlow displays. Useful for neatening up
    notebooks.
    ----------------------------------------------------------------------
    log_level = "0": all messages are logged (default behavior)
    log_level = "1": INFO messages are not printed
    log_level = "2": INFO and WARNING messages are not printed
    log_level = "3": INFO, WARNING, and ERROR messages are not printed
    """
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = log_level
