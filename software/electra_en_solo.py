#!/usr/bin/env python3

# %% [code]
import sys

sys.path.append("../trac2020_submission/src/")
import argparse
import pandas as pd
import os
import sys
import xml.etree.ElementTree as ET
import random
import numpy as np
import tqdm
from tqdm import notebook
from typing import Callable
import collections

# %% [code]
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

torch.__version__


# %% [code]
import utils.general as general_utils
import utils.transformer.data as transformer_data_utils
import utils.transformer.general as transformer_general_utils

# %% [code]
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics import accuracy_score


# %% [code]
def read_xml(xml_loc, lang):
    """reads an author's xml data
    
    Args:
        xml_loc : location of xml_file
        lang : either 'en' or 'es', for sanity check
    """
    tree = ET.parse(xml_loc)
    root = tree.getroot()

    assert root.attrib["lang"] == lang
    for c in root:
        for child in c:
            yield (child.text)


def create_author_to_document_df(data_paths, lang):
    """Creats a dict where author is mapped to his documents
    Args:
        data_paths Tuple[file_name, file_loc]
        lang: one of 'en' or 'es' for sanity check
    Returns:
        data_df: pd.DataFrame containing the author, his comments
    """
    author_to_doc = dict()

    author_list = []
    doc_list = []
    for file_name, file_loc in data_paths:
        author = file_name.split(".")[0]
        docs = read_xml(file_loc, lang)

        author_list.append(author)
        doc_list.append(list(docs))

    data_df = pd.DataFrame({"author": author_list, "doc_list": doc_list})
    return data_df


def read_truth_file(loc: str):
    """reads truth file
    Args:
        loc: full path to truth file
    Returns:
        truth_df : pd.DataFrame containing author,truth columns
    """

    with open(loc, "r") as file:
        lines = [line.strip().split(":::") for line in file]
    author_list = [i[0] for i in lines]
    truth = [int(i[1]) for i in lines]
    truth_df = pd.DataFrame({"author": author_list, "truth": truth})
    return truth_df


def make_fake_news_data_df(xml_data_dir, lang, test=False):
    """creates a pd.DataFrame of author, his comments, truth label"""
    data_paths = [
        (file, os.path.join(xml_data_dir, file))
        for file in os.listdir(xml_data_dir)
        if file[-1] != "t"
    ]

    author_documents_df = create_author_to_document_df(data_paths, lang)

    if not test:
        truth_path = [
            os.path.join(xml_data_dir, file)
            for file in os.listdir(xml_data_dir)
            if file[-1] == "t"
        ][0]

        truth_df = read_truth_file(truth_path)

        data_df = pd.merge(author_documents_df, truth_df, on="author")

        return data_df, author_documents_df, truth_df
    else:
        return author_documents_df


# %% [code]
def split_dataframe(df: pd.DataFrame, train_frac: float, shuffle: bool):
    """
    Splits DataFrame into train and val 
    Args:
        df: DataFrame to split, note: indexes will be reset
        train_frac: fraction to use for training 
        shuffle: Shuffles df if true
    Returns:
        split_df: DataFrame with splits mentioned in 'split' column
    """
    assert train_frac <= 1.0

    if train_frac == 1.0:
        df.split == "train"
        return df

    df.index = range(len(df.index))  # resetting index
    df = df.copy()

    if shuffle:
        df = df.sample(frac=1).sample(frac=1)

    val_frac = 1 - train_frac

    assert val_frac + train_frac == 1.0

    split_df = None

    labels = set(df.label)
    assert len(labels) != 1

    for lbl in labels:
        temp_df = df[df.label == lbl]
        _train_df = temp_df.sample(frac=train_frac)
        _train_df["split"] = "train"
        _val_df = temp_df[~temp_df.index.isin(_train_df.index)].copy()
        _val_df["split"] = "val"

        if split_df is None:
            split_df = pd.concat([_train_df, _val_df])
        else:
            split_df = pd.concat([split_df, _train_df, _val_df])

    # test that the the splits add up
    assert sum(df.label.value_counts()) == sum(
        split_df[split_df.split == "train"].label.value_counts()
    ) + sum(split_df[split_df.split == "val"].label.value_counts())

    return split_df


def join_and_add_tokens(doc: str, tokenizer):
    sep_token = f" {tokenizer.sep_token} "
    doc = sep_token.join(i for i in doc)
    doc = f"{tokenizer.cls_token} {doc}{sep_token}"

    return doc


def create_data_subset(frac: float, df: pd.DataFrame):
    labels = set(df.label)

    assert len(labels) != 1

    df = df.sample(frac=1).sample(frac=1)

    _dfs = []

    for lbl in labels:
        temp_df = df[df.label == lbl]
        _temp = temp_df.sample(frac=frac).copy()
        _dfs.append(_temp)

    mini_df = pd.concat(_dfs)

    return mini_df


# %% [code]
class TransformerClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        x = self.softmax(x)
        return x


class TransformerForSequenceClassification(nn.Module):
    def __init__(self, num_labels: int, model_name: str):
        """
        Args:
            num_labels: Number of labels to classify
            model_name: name of model to use
        """
        super().__init__()

        self.transformer = AutoModel.from_pretrained(model_name)
        self.transformer.config.num_labels = num_labels
        self.classifier = TransformerClassificationHead(self.transformer.config)

    def forward(self, input_ids=None, attention_mask=None):

        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        return logits


# %% [code]
class SimpleVectorizer:
    """Vectorizes Class to encode the samples into 
    their token ids and creates their respective attention masks
    """

    def __init__(self, tokenizer: Callable, max_seq_len: int):
        """
        Args:
            tokenizer (Callable): transformer tokenizer
            max_seq_len (int): Maximum sequence lenght 
        """
        self.tokenizer = tokenizer
        self._max_seq_len = max_seq_len

    def join_and_add_tokens(self, doc: str, tokenizer):
        sep_token = f" {tokenizer.sep_token} "
        doc = sep_token.join(i for i in doc)
        doc = f"{tokenizer.cls_token} {doc} {sep_token}"
        return doc

    def vectorize(self, doc_list, num_docs=14):
        """
        Randomly samples `num_docs` documents, concatenates and encodes them 
        Args:
            text: doc list to vectorize
        Returns:
            ids: Token ids of the 
            attn: Attention masks for ids 
        """
        if len(doc_list) < num_docs:
            num_docs = len(doc_list)

        text = self.join_and_add_tokens(
            random.sample(doc_list, num_docs), self.tokenizer
        )
        encoded = self.tokenizer.encode_plus(
            text,
            add_special_tokens=False,  # already added by preprocessor
            max_length=self._max_seq_len,
            pad_to_max_length=True,
        )
        #         ids =  np.array(encoded['input_ids'], dtype=np.int64)
        ids = torch.tensor(encoded["input_ids"], dtype=torch.long)

        return ids


# %% [code]
class FakeNewsDataset(Dataset):
    """PyTorch dataset class"""

    def __init__(
        self, data_df: pd.DataFrame, tokenizer: Callable, max_seq_length: int = None
    ):
        """
        Args:
            data_df (pandas.DataFrame): df containing the labels and text
            tokenizer (Callable): tokenizer for the transformer
            max_seq_length (int): Maximum sequece length to work with.
        """
        self.data_df = data_df
        self.tokenizer = tokenizer

        if max_seq_length is None:
            self._max_seq_length = self._get_max_len(data_df, tokenizer)
        else:
            self._max_seq_length = max_seq_length

        self.train_df = self.data_df[self.data_df.split == "train"]
        self.train_size = len(self.train_df)

        self.val_df = self.data_df[self.data_df.split == "val"]
        self.val_size = len(self.val_df)

        self.test_df = self.data_df[self.data_df.split == "test"]
        self.test_size = len(self.test_df)

        self._simple_vectorizer = SimpleVectorizer(tokenizer, self._max_seq_length)

        self._lookup_dict = {
            "train": (self.train_df, self.train_size),
            "val": (self.val_df, self.val_size),
            "test": (self.test_df, self.test_size),
        }

        self.set_split("train")

    def _get_max_len(self, data_df: pd.DataFrame, tokenizer: Callable):
        """Get the maximum lenght found in the data
        Args:
            data_df (pandas.DataFrame): The pandas dataframe with the data
            tokenizer (Callable): The tokenizer of the transformer
        Returns:
            max_len (int): Maximum length
        """
        len_func = lambda x: len(self.tokenizer.encode_plus(x)["input_ids"])
        max_len = data_df.text.map(len_func).max()

        return max_len

    def set_split(self, split="train"):
        """selects the splits in the dataset using a column in the dataframe """
        self._target_split = split
        self._target_df, self._target_size = self._lookup_dict[split]

    def __len__(self):
        return self._target_size

    def __getitem__(self, index):
        """the primary entry point method for PyTorch datasets
        
        Args:
            index (int): the index to the data point 
        Returns:
            a dictionary holding the data point's features (x_data) and label (y_target)
        """
        row = self._target_df.iloc[index]

        indices = self._simple_vectorizer.vectorize(row.doc_list)

        label = row.label

        return {
            "x_author": row.author,
            "x_data": indices,
            "x_index": index,
            "y_target": label,
        }

    def get_num_batches(self, batch_size):
        """Given a batch size, return the number of batches in the dataset
        
        Args:
            batch_size (int)
        Returns:
            number of batches in the dataset
        """
        return len(self) // batch_size


# %% [code]
def generate_batches(
    dataset,
    batch_size,
    shuffle=True,
    drop_last=False,
    device="cpu",
    pinned_memory=False,
    n_workers=0,
):
    """
    A generator function which wraps the PyTorch DataLoader. It will 
      ensure each tensor is on the write device location.
    """
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        pin_memory=pinned_memory,
    )

    for data_dict in dataloader:
        out_data_dict = {}
        out_data_dict["x_data"] = data_dict["x_data"].to(
            device, non_blocking=(True if pinned_memory else False)
        )
        out_data_dict["x_index"] = data_dict["x_index"]
        out_data_dict["y_target"] = data_dict["y_target"].to(
            device, non_blocking=(True if pinned_memory else False)
        )
        out_data_dict["x_author"] = data_dict["x_author"]
        yield out_data_dict


# %% [code]
def run_model_on_test(model, dataset, state_dict):
    """
    Args:
        mode: training mode if true else eval
        model: torch.nn
    """

    dataset.set_split("val")

    mode = "val"
    total = dataset.get_num_batches(args.batch_size)

    # pbar = tqdm.tqdm(desc=f"mode={mode}", total=total, position=0, leave=True,)

    model.eval()

    batch_generator = generate_batches(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False,
        device=args.device,
        drop_last=False,
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    running_acc = 0.0

    for batch_index, batch_dict in enumerate(batch_generator):

        y_pred = model(input_ids=batch_dict["x_data"])

        y_pred = y_pred.view(-1, len(set(dataset.data_df.label)))

        y_pred = y_pred.detach().cpu()

        batch_dict["y_target"] = batch_dict["y_target"].cpu()

        acc_t = transformer_general_utils.compute_accuracy(
            y_pred, batch_dict["y_target"]
        )

        state_dict["batch_preds"].append(y_pred)
        state_dict["batch_targets"].append(batch_dict["y_target"])
        state_dict["batch_indexes"].append(batch_dict["x_index"])
        state_dict["batch_authors"].extend(batch_dict["x_author"])

        running_acc += (acc_t - running_acc) / (batch_index + 1)

        # pbar.set_postfix(running_acc=running_acc)

        # pbar.update()

    state_dict[f"{mode}_preds"].append(torch.cat(state_dict["batch_preds"]).cpu())
    state_dict["authors"].append(state_dict["batch_authors"])
    state_dict[f"{mode}_targets"].append(torch.cat(state_dict["batch_targets"]).cpu())
    state_dict[f"{mode}_indexes"].append(torch.cat(state_dict["batch_indexes"]).cpu())
    acc = transformer_general_utils.compute_accuracy(
        state_dict[f"{mode}_preds"][-1], state_dict[f"{mode}_targets"][-1]
    )

    state_dict[f"{mode}_accuracies"].append(acc)

    state_dict["batch_preds"] = []
    state_dict["batch_targets"] = []
    state_dict["batch_indexes"] = []
    state_dict["batch_authors"] = []

    # pbar.set_postfix(running_acc=acc)

    return state_dict, acc


# %% [code]
def majority_voting(ts, test=False):
    _val_targets = ts["val_targets"][0]

    prev_acc = 0

    all_preds = []

    for i in range(len(ts["val_preds"])):

        _val_preds = ts["val_preds"][i]

        if not test:
            curr_acc = accuracy_score(
                y_pred=_val_preds.argmax(axis=1), y_true=_val_targets
            )
            print(curr_acc)

        all_preds.append(_val_preds.argmax(axis=1))

    ballot = np.column_stack(all_preds)
    pred = [collections.Counter(l).most_common(1)[0][0] for l in ballot.tolist()]
    if not test:
        acc = accuracy_score(y_pred=pred, y_true=_val_targets)
        return acc
    else:
        return pred


# %% [code]

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--indir", help="Input directory", required=True)
    parser.add_argument("-o", "--outdir", help="Output directory", required=True)
    parser.add_argument(
        "-m", "--bestmodeldir", help="full path to a .pt file", required=True
    )

    in_args = vars(parser.parse_args())

    indir = in_args["indir"]

    endir = os.path.join(indir, "en")
    esdir = os.path.join(indir, "es")

    outdir = in_args["outdir"]

    newpath = os.path.join(outdir, "en")

    if not os.path.exists(newpath):
        os.makedirs(newpath)

    args = argparse.Namespace(
        en_data_dir=endir,
        es_data_dir=esdir,
        device=("cuda" if torch.cuda.is_available() else "cpu"),
        batch_size=20,
        best_model=in_args["bestmodeldir"],
        output_dir=newpath,
        lang="en",
    )

    print("Making Dataframe")
    en_data_df = make_fake_news_data_df(args.en_data_dir, args.lang, True)
    en_data_df["label"] = (np.random.rand(len(en_data_df)) > 0.5).astype(int)

    en_data_df["split"] = "val"

    tokenizer = AutoTokenizer.from_pretrained("google/electra-small-discriminator")

    print("Making Dataset")

    dataset = FakeNewsDataset(en_data_df, tokenizer, 512)

    # %% [code]
    print("Loading Model")

    model = TransformerForSequenceClassification(
        2, "google/electra-small-discriminator"
    )
    model.load_state_dict(torch.load(args.best_model))
    model.to(args.device)

    train_state = (
        general_utils.make_train_state()
    )  # dictionary for saving training routine information
    train_state["authors"] = []
    train_state["batch_authors"] = []

    print("Starting Predictions")

    for i in range(15):
        with torch.no_grad():
            train_state, val_acc = run_model_on_test(model, dataset, train_state)
        print(f"Finished iter : {i} ")

    # %% [code]

    print(f"Model Predictions Done")
    # %% [code]
    test_preds = majority_voting(train_state, True)
    test_authors = train_state["authors"][0]

    print(f"Voting Complete!")

    # %% [code]
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    print(f"Writing Predictions to outdir:")

    for aut, lbl in zip(test_authors, test_preds):
        print(aut, lbl)
        file_name = f"{aut}.xml"
        with open(os.path.join(args.output_dir, file_name), "w") as file:
            file.write(f'<author id="{aut}"\n')
            file.write(f'\tlang="{args.lang}"\n')
            file.write(f'\ttype="{lbl}"\n')
            file.write("/>\n")

    print("Finished!")
