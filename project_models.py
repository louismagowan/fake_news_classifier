# IMPORTS
import pandas as pd
import numpy as np
from tensorflow.keras.layers import Dense, BatchNormalization, Reshape, Activation
from tensorflow.keras.layers import Embedding, GRU, Bidirectional
from tensorflow.keras import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.initializers import Constant
import spacy
from sklearn.metrics import f1_score, roc_auc_score


############ ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# GENERAL

def pred_labels(model, X_test):

    """
    Function to make predictions for given X data and instead of
    returning raw probabilities, returns the predictions in label
    form: 0 = true news, 1 = fake news.
    ----------------------------------------------------------------------
    model =  a trained model to predict with
    X_test = array of features you want to predict from
    """

    # Make predictions / get raw probabilities
    raw_preds = model.predict(X_test)

    # Convert probabilities into labels
    preds = (raw_preds > 0.5).astype(int).reshape(-1)

    return preds


def pred_text_as_labels(text, tokenizer,
    model, maxlen = 500,
     truncating = "post",
     as_labels = True):
    """
    Function to make predictions for new texts/articles using a fitted tokenizer
    and a fitted model. Returns either label predictions (0/1) or probabilities
    for whether the articles are true/fake, depending on as_labels arg.
    ----------------------------------------------------------------------
    text = list of strings, texts/articles you want to try and classify
    tokenizer = a fitted Keras Tokenizer() object
    model =  a trained model to predict with
    max_len = int, maximum length of sequences
    padding = str, whether sequences should be padded at the front or back
    truncating = str, whether sequences should be truncated at the front or back
    as_labels = bool, if True will return model predictions in label form- 0 = true and
    1 = fake. If False, gives the raw probabilities.
    """

    # Convert text to sequences
    seqs = tokenizer.texts_to_sequences(text)
    # Pad/trim sequences
    padded_seqs = pad_sequences(seqs, maxlen = maxlen, truncating = truncating)
    # Make predictions
    raw_preds = model.predict(padded_seqs)

    if as_labels:
        # Convert from probabilities into labels
        preds = (raw_preds > 0.5).astype(int).reshape(-1)
        return preds

    else:
        return raw_preds


def get_test_metrics(model, X_test, y_test,
                     history,
                    embedding = None,
                    batch_normalize = False,
                    verbose = 0):
    """
    Evaluates a fitted model against test data in terms of accuracy,
    ROC AUC score and F1 score. Must be called after the embedding layer,
    regularize toggle and batch_normalize toggle have all been instantiated.
    Returns a df of model name, specs and test metrics.
    ----------------------------------------------------------------------
    model =  a trained model to predict with
    X_test = array of padded text sequences
    y_test = int, 0/1 labels for true/fake news
    all_results = pd.DataFrame, either empty or containing rows of
    other models' results
    history = Keras history object
    embedding = str/None, name of embedding layer used in embedding dict
    batch_normalize = bool, whether or not model was batch normalized
    verbose = int, controls messaging of model.evaluate
    """
    
    # Get number of epochs run for
    n_epochs = len(history.history["loss"])

    # Get test accuracy
    test_acc = model.evaluate(X_test, y_test, verbose = verbose)[1]

    # Get raw predictions / probabilities for ROC AUC
    probs = model.predict(X_test)
    test_roc_auc = roc_auc_score(y_test, probs)

    # Get label predictions for F1
    preds = pred_labels(model, X_test)
    test_f1 = f1_score(y_test, preds)

    # Save results
    results = pd.DataFrame({"embedding":embedding,
                            "batch_normalize":batch_normalize,
                            "accuracy":test_acc,
                           "roc_auc":test_roc_auc, "f1":test_f1,
                           "epochs":n_epochs}, index = [0])

    return results


############ ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Embeddings


def spacy_embedding(tokenizer, maxlen = 500, show_progress = False):

    """Function to create SpaCy embedding layer. Uses the en_core_web_sm pipeline,
    a small English-language pipeline appropriate for blogs, news and comments.
    This takes a while to run.
    ----------------------------------------------------------------------
    tokenizer = a fitted Keras Tokenizer() object
    max_len = int, maximum length of sequences
    show_progress = bool, simple indicator to tell you how much progress with
    the embedding you have made as %.
    """

    # Load the spacy pipeline
    # small English pipeline trained on written web text (blogs, news, comments)
    nlp = spacy.load("en_core_web_sm")
    # Get vocab size of tokenizer
    vocab_size = len(tokenizer.word_index) + 1

    # Get the number of embedding dimensions SpaCy uses
    embedding_dim = nlp("any_word").vector.shape[0]
    # Create a matrix to use in embedding layer
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    # Iterate through our vocabulary, mapping words to spacy embedding
    # this will take a while to run
    for i, word in enumerate(tokenizer.word_index):
        embedding_matrix[i] = nlp(word).vector
        # Show progress if desired
        if show_progress:
            if i % 10000 == 0 and i > 0:
                print(round(i*100/vocab_size, 3), "% complete")


    # Load the embedding matrix as the weights matrix for the embedding layer
    # Set trainable to False as the layer is already "learned"
    Embedding_layer = Embedding(
        vocab_size,
        embedding_dim,
        input_length = maxlen,
        embeddings_initializer=Constant(embedding_matrix),
        trainable=False,
        name = "spacy_embedding")

    return Embedding_layer



def keras_embedding(tokenizer, embedding_dim = 256, maxlen = 500):

    """Function to create a custom Keras embedding layer.
    ----------------------------------------------------------------------
    tokenizer = a fitted Keras Tokenizer() object
    max_len = int, maximum length of sequences
    """

    # Get vocab size of tokenizer
    vocab_size = len(tokenizer.word_index) + 1

    # Load the embedding matrix as the weights matrix for the embedding layer
    # Set trainable to False as the layer is already "learned"
    Embedding_layer = Embedding(
        vocab_size,
        embedding_dim,
        input_length = maxlen,
        name = "keras_embedding")

    return Embedding_layer


############ ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# GRU

def bi_gru(loss = "binary_crossentropy",
                optimizer = "adam",
               metrics = ["accuracy"], 
               batch_normalize = False,
               embedding = None,
               maxlen = 500,
               hidden_dense_units = 256,
               dense_kernel_initializer = "glorot_uniform",
               rnn_units = 32,
               rnn_kernel_initializer = "glorot_uniform"):
    """
    Creates a GRU model designed to be used with the text data only.
    Can be built with either Keras, SpaCy, GloVe or no embedding.
    Returns a compiled Keras model of a bidirectional GRU (32*2) and 2 Dense layers (256, 1).
    There is an option to include a batch normalisation layer too.
    ----------------------------------------------------------------------
    loss = str, name of loss function to use
    optimizer = Keras optimizer, set to 'adam' but any optimizer can be passed
    metrics =  list of Keras metrics to use to evaluate with
    batch_normalize = bool, if True adds batch normalisation between hidden Dense
    and output layer.
    embedding = None/Keras embedding instance: The type of embedding to use (SpaCy or Keras).
    maxlen = int, shape of input (length of sequences)
    hidden_dense_units = int, number of hidden units in the hidden dense layer
    dense_kernel_initializer = str or keras.initializers object for the weights
    of the Dense layer.
    rnn_units = int, number of hidden units in the recurrent computation of GRU.
    rnn_kernel_initializer = str or keras.initializers object for the weights
    of the GRU layer.
    """

    # Build model
    model = Sequential(name = "GRU")

    # Add embedding if desired
    if embedding:
        # Embedding contains input shape
        model.add(embedding)
    else:
        # Otherwise reshape data to work with GRU
        model.add(Reshape((maxlen, 1), input_shape = (maxlen, ), name = "Reshaping"))

    # Add GRU
    model.add(Bidirectional(GRU(rnn_units,
                                kernel_initializer = rnn_kernel_initializer),
                                name = "Bidirectional_GRU"))

    # Baseline model
    model.add(Dense(hidden_dense_units, name = "Linear_Dense",
                    kernel_initializer = dense_kernel_initializer))

    # Batch normalised model
    if batch_normalize:
        model.add(BatchNormalization(name = "Batch_Norm1"))

    # Apply non-linear activation, specified in this way to be consistent
    # with the original paper
    model.add(Activation("relu", name = "ReLU_Activation"))

    # Output layer
    model.add(Dense(1, activation = "sigmoid", name = "Output",
                    kernel_initializer = dense_kernel_initializer))
    # Compile model
    model.compile(loss = loss, optimizer = optimizer,
                  metrics = metrics)

    return model
