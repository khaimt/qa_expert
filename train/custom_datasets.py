# This script is created based on: https://github.com/MeetKai/functionary/blob/main/functionary/train/custom_datasets.py
import datetime
import json
import os
import pickle
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import transformers
from torch.utils.data import Dataset
from qa_expert import prompt_utils
from gen_data import utility


def map_raw_data_to_input_dic(raw_data: List[Dict], tokenizer: Any, padding: str, batch_size: int = 5000) -> List[Dict]:
    invalid_count = 0
    data_size = len(raw_data)
    data_points = []
    t1 = datetime.datetime.now()
    for start, end in utility.get_batch_indices(data_size, batch_size):
        batch_messages = [prompt_utils.convert_multi_qa_format_to_messages(item) for item in raw_data[start:end]]
        batch_result = prompt_utils.preprare_training_inputs_batch(batch_messages, tokenizer, padding)
        assert len(batch_result) == len(raw_data[start:end])
        for item in batch_result:
            if is_valid_labels(item["labels"]):
                data_points.append(item)
            else:
                print("invalid: ")
                invalid_count += 1
        t2 = datetime.datetime.now()
        avg_time = (t2 - t1).total_seconds() / len(data_points)
        remaining_time = avg_time * (data_size - len(data_points))
        print(
            f"{len(data_points)}/{data_size}, avg_time per 1000 data points: {avg_time * 1000}, remaining time: {remaining_time}"
        )
    if invalid_count > 0:
        print(f"*****WARNING: invalid data points: {invalid_count} because of labels=-100 all the time")
    assert len(data_points) == data_size - invalid_count
    return data_points


def merge_data_points_by_length(lengths: List[int], max_length: int) -> List[List[int]]:
    """given lengths of data points, we merge them into groups such that the sum of lengths
    in each group is less than max_length. This is known as: https://en.wikipedia.org/wiki/Bin_packing_problem
    Here is the greedy algorithm
    Args:
        lengths (List[int]): _description_
        max_length (int): _description_

    Returns:
        _type_: groups of indices: [[index1, index2, ...], [], ...]
    """
    items = [{"length": length, "index": i} for i, length in enumerate(lengths)]
    items = sorted(items, key=lambda x: x["index"])
    merges = []
    current_sum = 0
    current_list = []
    for i in range(len(items)):
        cur_length = items[i]["length"]
        if cur_length + current_sum <= max_length:
            current_sum += items[i]["length"]
            current_list.append(i)
        else:
            merges.append(current_list)
            current_list = [i]
            current_sum = cur_length
    if len(current_list) > 0:
        merges.append(current_list)
    result = []
    for merge in merges:
        sub_items = [items[index]["index"] for index in merge]
        result.append(sub_items)
    return result


def get_causal_mask(length: int, sliding_window: Optional[int] = None):
    """
    Make causal mask used for sliding window attention
    """
    tensor = torch.full(
        (length, length),
        fill_value=1,
    )
    mask = torch.tril(tensor, diagonal=0)
    # make the mask banded to account for sliding window
    if sliding_window is not None:
        mask = torch.triu(mask, diagonal=-sliding_window)
    mask = torch.log(mask)
    return mask


def create_mask_padding_right(
    lengths: List[int], model_max_length: int, sliding_window: Optional[int] = None
) -> torch.tensor:
    """create attention_mask: N x N where masked value = m_value
    Args:
        lengths (List[int]): length of data points
        tokenizer (Any): _description_
        m_value (float): _description_

    Returns:
        torch.tensor: _description_
    """
    result = torch.full((model_max_length, model_max_length), float("-inf"))
    acc_leng = 0
    for length in lengths:
        # mask for a data point with length
        x = get_causal_mask(length, sliding_window)
        result[acc_leng : acc_leng + length, acc_leng : acc_leng + length] = x
        acc_leng += length
    pad_length = model_max_length - sum(lengths)
    if pad_length > 0:
        result[-pad_length:, :] = 0
        result[:, -pad_length:] = float("-inf")
    return result


def create_mask_padding_left(
    lengths: List[int], model_max_length: int, sliding_window: Optional[int] = None
) -> torch.tensor:
    result = torch.full((model_max_length, model_max_length), float("-inf"))
    pad_length = model_max_length - sum(lengths)
    acc_leng = 0
    for length in [pad_length] + lengths:
        x = get_causal_mask(length, sliding_window)
        result[acc_leng : acc_leng + length, acc_leng : acc_leng + length] = x
        acc_leng += length
    return result


def create_mask_from_lengths(lengths: List[int], tokenizer: Any, sliding_window: Optional[int] = None) -> torch.tensor:
    if tokenizer.padding_side == "left":
        return create_mask_padding_left(lengths, tokenizer.model_max_length, sliding_window)
    return create_mask_padding_right(lengths, tokenizer.model_max_length, sliding_window)


def merge_data_points(data_points: List[Dict], tokenizer: Any, sliding_window: Any) -> Dict:
    input_ids = []
    lengths = []
    label_ids = []
    for item in data_points:
        input_ids += item["input_ids"]
        # assert item["labels"][0] == -100 # This is to make sure that the first token won't be included in computing loss
        labels = list(item["labels"])
        labels[0] = -100
        label_ids += labels
        lengths.append(len(item["input_ids"]))
    attention_mask = create_mask_from_lengths(lengths, tokenizer, sliding_window)
    pad_leng = tokenizer.model_max_length - len(input_ids)  # padding to model_max_length
    if tokenizer.padding_side == "right":
        input_ids = input_ids + [tokenizer.pad_token_id for _ in range(pad_leng)]
        label_ids = label_ids + [-100 for _ in range(pad_leng)]
    else:
        input_ids = [tokenizer.pad_token_id for _ in range(pad_leng)] + input_ids
        label_ids = [-100 for _ in range(pad_leng)] + label_ids
    assert len(input_ids) == len(label_ids) == attention_mask.size(0)
    return {
        "input_ids": torch.tensor(input_ids),
        "labels": torch.tensor(label_ids),
        "attention_mask": torch.unsqueeze(attention_mask, 0),  # unsqueeze <-- because the shape is: B x 1 x N x N
    }


def is_valid_labels(labels: Union[List[int], torch.Tensor]) -> bool:
    """by setting max_length, there might be the case that the labels are all -100 -> loss=nan
    Args:
        labels (Union[List[int], torch.Tensor]): _description_

    Returns:
        bool: _description_
    """
    if type(labels) is list:
        non_mask_count = 0
        for label in labels:
            if label != -100:
                non_mask_count += 1
        if non_mask_count == 0:
            return False
        return True
    elif type(labels) is torch.tensor:
        if sum(labels + 100) == 0:  # mypy: ignore-errors
            return False
        return True
    return True


def remove_invalid_label_items(data_points: List[Dict]) -> List[Dict]:
    """Remove data points where labels are all -100

    Args:
        data_points (List[Dict]): _description_

    Returns:
        _type_: _description_
    """
    result = []
    for dp in data_points:
        if is_valid_labels(dp["labels"]):
            result.append(dp)
    return result


class CachedDataset(Dataset):
    def __init__(self, tokenizer: Any, cached_folder: Optional[str] = None, ignore_cached: bool = False) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.data_points: List[Dict] = []
        self.load_from_cache = False
        if cached_folder is not None and not ignore_cached:
            data_path = self.get_data_point_path(cached_folder)
            if os.path.exists(data_path):
                print(f"cached found, load from cached: {cached_folder}")
                self.load(cached_folder)
                self.load_from_cache = True

    def __len__(self):
        return len(self.data_points)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return self.data_points[i]

    def create_meta_info(self):
        return {"max_length": self.tokenizer.model_max_length, "size": len(self.data_points)}

    def load(self, folder: str):
        t1 = datetime.datetime.now()
        with open(self.get_data_point_path(folder), "rb") as file:
            self.data_points = pickle.load(file)
        t2 = datetime.datetime.now()
        print("time for loading cached data: ", (t2 - t1).total_seconds())

    def get_data_point_path(self, folder: str) -> str:
        return os.path.join(folder, "data_points.pkl")

    def get_metainfo_path(self, folder: str) -> str:
        return os.path.join(folder, "meta_info.json")

    def dump(self, folder: str):
        t1 = datetime.datetime.now()
        if not os.path.exists(folder):
            os.mkdir(folder)

        with open(self.get_data_point_path(folder), "wb") as file:
            pickle.dump(self.data_points, file)

        with open(self.get_metainfo_path(folder), "w") as f:
            f.write(json.dumps(self.create_meta_info()))
        t2 = datetime.datetime.now()
        print("time for dumping data: ", (t2 - t1).total_seconds())

    def stat(self):
        print(json.dumps(self.create_meta_info()))


class CustomDataset(CachedDataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        raw_data: List[Dict],
        tokenizer: transformers.PreTrainedTokenizer,
        cached_folder: Optional[str] = None,
        ignore_cached: bool = False,
        batch_size: int = 5000,
        **kwargs,
    ):
        super().__init__(tokenizer, cached_folder, ignore_cached)

        if not self.load_from_cache:  # if not loaded from cached
            self.data_points = map_raw_data_to_input_dic(
                raw_data, tokenizer, padding="max_length", batch_size=batch_size
            )
            if cached_folder is not None:
                print(f"dump data to cached: {cached_folder}")
                self.dump(cached_folder)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        dp = self.data_points[i]
        result = {}
        for key in dp:
            result[key] = torch.tensor(dp[key])
        return result


class PackedDataset(CachedDataset):
    def __init__(
        self,
        raw_data: List[Dict],
        tokenizer: transformers.PreTrainedTokenizer,
        cached_folder: Optional[str] = None,
        ignore_cached: bool = False,
        batch_size: int = 5000,
        **kwargs,
    ):
        super().__init__(tokenizer, cached_folder, ignore_cached)
        self.sliding_window = kwargs.get("sliding_window", None)
        if not self.load_from_cache:
            self.data_points = map_raw_data_to_input_dic(
                raw_data, tokenizer, padding="do_not_pad", batch_size=batch_size
            )
            self.update_packing_info()
            if cached_folder is not None:
                print(f"dump data to cached: {cached_folder}")
                self.dump(cached_folder)
        else:  # update packing
            self.update_packing_info()

    def update_packing_info(self):
        self.lengths = [len(item["input_ids"]) for item in self.data_points]
        self.groups = merge_data_points_by_length(self.lengths, self.tokenizer.model_max_length)

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        group = self.groups[i]
        group_data_points = [self.data_points[index] for index in group]
        return merge_data_points(group_data_points, self.tokenizer, self.sliding_window)

    def stat(self):
        print(f"number of original data points:{len(self.data_points)}; packed to: {len(self.groups)} data points")
        original_avg_length = sum(self.lengths) / len(self.lengths)
        packed_lengths = []
        for group in self.groups:
            lengths = [self.lengths[index] for index in group]
            packed_lengths.append(sum(lengths))
        avg_packed_length = sum(packed_lengths) / len(packed_lengths)
        print(f"original avg length: {original_avg_length}; avg packed length: {avg_packed_length}")


def pack_data_points_FA(data_points: List[Dict], tokenizer: Any) -> Dict:
    input_ids = []
    lengths = []
    label_ids = []
    attention_mask = []
    for index, item in enumerate(data_points):
        input_ids += item["input_ids"]
        # assert item["labels"][0] == -100 # This is to make sure that the first token won't be included in computing loss
        labels = list(item["labels"])
        labels[0] = -100
        label_ids += labels
        lengths.append(len(item["input_ids"]))
        attention_mask += [index + 1 for _ in range(len(item["input_ids"]))]

    pad_leng = tokenizer.model_max_length - len(input_ids)  # padding to model_max_length
    if tokenizer.padding_side == "right":
        input_ids = input_ids + [tokenizer.pad_token_id for _ in range(pad_leng)]
        label_ids = label_ids + [-100 for _ in range(pad_leng)]
        attention_mask = attention_mask + [0 for _ in range(pad_leng)]
    else:
        input_ids = [tokenizer.pad_token_id for _ in range(pad_leng)] + input_ids
        label_ids = [-100 for _ in range(pad_leng)] + label_ids
        attention_mask = [0 for _ in range(pad_leng)] + attention_mask

    assert len(input_ids) == len(label_ids) == len(attention_mask)
    return {
        "input_ids": torch.tensor(input_ids),
        "labels": torch.tensor(label_ids),
        "attention_mask": torch.tensor(attention_mask),  # unsqueeze <-- because the shape is: B x 1 x N x N
    }


class FAPackedDataset(PackedDataset):
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        group = self.groups[i]
        group_data_points = [self.data_points[index] for index in group]
        return pack_data_points_FA(group_data_points, self.tokenizer)
