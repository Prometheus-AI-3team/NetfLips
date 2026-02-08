import numpy as np
import torch
from fairseq.data import FairseqDataset, data_utils
from typing import Dict, List, Optional, Tuple, Union
# from fairseq.data.language_pair_dataset
from fairseq.data.audio.speech_to_speech_dataset import SpeechToSpeechDataset
from fairseq.data.denoising_dataset import DenoisingDataset
from fairseq.data import ConcatDataset, Dictionary

from fairseq.data.audio.speech_to_text_dataset import SpeechToTextDatasetCreator, SpeechToTextDataset
from fairseq.data import (
    AppendTokenDataset,
    ConcatDataset,
    DenoisingDataset,
    Dictionary,
    PrependTokenDataset,
    ResamplingDataset,
    SortDataset,
    TokenBlockDataset,
    data_utils,
)

import os
import math

def process_units(units, reduce=False):
    if not reduce:
        return units

    out = [u for i, u in enumerate(units) if i == 0 or u != units[i - 1]]
    return out

class UnitToUnitDataset(FairseqDataset):
    def __init__(
        self,
        split: str,
        is_train_split: bool,
        src_unit_paths: List[str],
        src_n_units: List[int],
        tgt_unit_paths: List[str],
        tgt_n_units: List[int],
        src_langs: Optional[List[str]] = None,
        tgt_langs: Optional[List[str]] = None,
        src_dict: Dictionary = None,
        tgt_dict: Dictionary = None,
    ):

        self.split, self.is_train_split = split, is_train_split
        self.src_unit_paths, self.src_n_units = src_unit_paths, src_n_units
        self.tgt_unit_paths, self.tgt_n_units = tgt_unit_paths, tgt_n_units
        self.n_samples = len(src_unit_paths)
        assert len(src_n_units) == self.n_samples > 0
        assert len(tgt_unit_paths) == self.n_samples
        assert len(tgt_n_units) == self.n_samples
        assert src_langs is None or len(src_langs) == self.n_samples
        assert tgt_langs is None or len(tgt_langs) == self.n_samples
        self.src_langs, self.tgt_langs = src_langs, tgt_langs
        self.src_dict, self.tgt_dict = src_dict, tgt_dict

        self.offset = 4

    def __len__(self):
        return self.n_samples

    @property
    def sizes(self):
        return np.array(self.src_n_units)

    def tgt_sizes(self):
        return np.array(self.tgt_n_units)

    def __getitem__(self, index: int):
        src_unit_path = self.src_unit_paths[index]
        tgt_unit_path = self.tgt_unit_paths[index]

        try:
            source = self.src_dict.encode_line(
                " ".join(map(lambda x: str(x), process_units(torch.load(src_unit_path), reduce=True))),
                add_if_not_exist=False,
                append_eos=True,
            ).long()
        except:
            source = self.src_dict.encode_line(
                " ".join(map(lambda x: str(x), process_units(np.array([int(x) for x in open(src_unit_path).readline().strip().split(" ")]), reduce=True))),
                add_if_not_exist=False,
                append_eos=True,
            ).long()

        try:
            target = self.tgt_dict.encode_line(
                " ".join(map(lambda x: str(x), process_units(torch.load(tgt_unit_path), reduce=True))),
                add_if_not_exist=False,
                append_eos=True,
            ).long()
        except:
            target = self.tgt_dict.encode_line(
                " ".join(map(lambda x: str(x), process_units(np.array([int(x) for x in open(tgt_unit_path).readline().strip().split(" ")]), reduce=True))),
                add_if_not_exist=False,
                append_eos=True,
            ).long()

        # assert len(source) - 1 == self.src_n_units[index], f'{len(source)}, {self.src_n_units[index]}, {self.src_unit_paths[index]}'
        # assert len(target) - 1 == self.tgt_n_units[index], f'{len(target)}, {self.tgt_n_units[index]}, {self.tgt_unit_paths[index]}'

        # assert self.src_n_units[index] <= 1024, f'src length: {self.src_n_units[index]}'
        # assert self.tgt_n_units[index] <= 1024, f'tgt length:{self.tgt_n_units[index]}'

        return source, target

class UnitToUnitDatasetCreator(object):
    # mandatory columns
    KEY_SRC_UNIT_PATH, KEY_SRC_N_UNITS = "src_unit_path", "src_n_units"
    KEY_TGT_UNIT_PATH, KEY_TGT_N_UNITS = "tgt_unit_path", "tgt_n_units"
    # optional cdolumns
    KEY_SRC_LANG, KEY_TGT_LANG = "src_lang", "tgt_lang"
    # default values
    DEFAULT_LANG = ""

    @classmethod
    def _from_list(
        cls,
        split_name: str,
        is_train_split,
        samples: List[Dict],
        src_dict: Dictionary = None,
        tgt_dict: Dictionary = None,
        src_lang: str = "",
        tgt_lang: str = "",
        max_target_positions: int = 1024,
    ) -> UnitToUnitDataset:
        samples = [s for s in samples if 0 < int(s[cls.KEY_TGT_N_UNITS]) <= max_target_positions - 2 and 0 < int(s[cls.KEY_SRC_N_UNITS]) <= max_target_positions - 2]

        src_unit_paths = [
            s[cls.KEY_SRC_UNIT_PATH] for s in samples
        ]
        tgt_unit_paths = [
            s[cls.KEY_TGT_UNIT_PATH] for s in samples
        ]
        src_n_units = [int(s[cls.KEY_SRC_N_UNITS]) for s in samples]
        tgt_n_units = [int(s[cls.KEY_TGT_N_UNITS]) for s in samples]
        src_langs = [s.get(cls.KEY_SRC_LANG, src_lang) for s in samples]
        tgt_langs = [s.get(cls.KEY_TGT_LANG, tgt_lang) for s in samples]

        ds = UnitToUnitDataset(
            split=split_name,
            is_train_split=is_train_split,
            src_unit_paths=src_unit_paths,
            src_n_units=src_n_units,
            tgt_unit_paths=tgt_unit_paths,
            tgt_n_units=tgt_n_units,
            src_langs=src_langs,
            tgt_langs=tgt_langs,
            src_dict=src_dict,
            tgt_dict=tgt_dict,
        )
        return ds

    @classmethod
    def from_tsv(
        cls,
        root: str,
        split: str,
        is_train_split: bool,
        epoch: int,
        seed: int,
        src_dict: Dictionary = None,
        tgt_dict: Dictionary = None,
        max_target_positions: int = 1024,
    ) -> SpeechToSpeechDataset:
        datasets = []
        with open(os.path.join(root, split, 'lang_pairs.txt')) as f:
            lang_pairs = [l.strip() for l in f.readlines()]
        for lang_pair in lang_pairs:
            samples = SpeechToTextDatasetCreator._load_samples_from_tsv(root, os.path.join(split, lang_pair))
            src_lang, tgt_lang = lang_pair.split('/')[-2:]
            ds = cls._from_list(
                split_name=split,
                is_train_split=is_train_split,
                samples=samples,
                src_dict=src_dict,
                tgt_dict=tgt_dict,
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                max_target_positions=max_target_positions,
            )
            datasets.append(ds)
        return MyConcatDataset(datasets) # if len(datasets) > 1 else datasets[0]

class MyConcatDataset(ConcatDataset):
    def __init__(self, datasets, sample_ratios=1):
        super().__init__(datasets, sample_ratios)
        self.src_langs = sum([d.src_langs for d in self.datasets], [])
        self.tgt_langs = sum([d.tgt_langs for d in self.datasets], [])
        self.src_unit_paths = sum([d.src_unit_paths for d in self.datasets], [])
        self.tgt_unit_paths = sum([d.tgt_unit_paths for d in self.datasets], [])

    @property
    def tgt_sizes(self):
        _dataset_sizes = []
        for ds, sr in zip(self.datasets, self.sample_ratios):
            if isinstance(ds.sizes, np.ndarray):
                _dataset_sizes.append(np.tile(ds.tgt_sizes(), sr))
            else:
                # Only support underlying dataset with single size array.
                assert isinstance(ds.tgt_sizes(), list)
                _dataset_sizes.append(np.tile(ds.tgt_sizes()[0], sr))
        return np.concatenate(_dataset_sizes)

class MyPrependTokenDataset(PrependTokenDataset):
    def __init__(self, dataset, token=None):
        super().__init__(dataset, token)
        self.src_langs = dataset.src_langs
        self.tgt_langs = dataset.tgt_langs
        if token is not None:
            self._tgt_sizes = np.array(dataset.tgt_sizes) + 1
        else:
            self._tgt_sizes = dataset.tgt_sizes

    @property
    def tgt_sizes(self):
        return self._tgt_sizes

    def __getitem__(self, idx):
        source, target = self.dataset[idx]
        if self.token is not None:
            source = torch.cat([source.new([self.token]), source])
            target = torch.cat([target.new([self.token]), target])
        return source, target
    
class MyAppendTokenDataset(AppendTokenDataset):
    def __init__(self, dataset, token=None, replace_with_lang_token=False, dictionary=None):
        super().__init__(dataset, token)
        self.replace_with_lang_token = replace_with_lang_token
        self.dictionary = dictionary
        self.src_langs = dataset.src_langs
        self.tgt_langs = dataset.tgt_langs
        if token is not None:
            self._tgt_sizes = np.array(dataset.tgt_sizes) + 1
        else:
            self._tgt_sizes = dataset.tgt_sizes
            
    @property
    def tgt_sizes(self):
        return self._tgt_sizes

    def __getitem__(self, idx):
        source, target = self.dataset[idx]
        if self.token is not None:
            if self.replace_with_lang_token:
                src_lang = self.src_langs[idx]
                tgt_lang = self.tgt_langs[idx]

                source = torch.cat([source, source.new([self.dictionary.index("[{}]".format(src_lang))])])
                target = torch.cat([target, target.new([self.dictionary.index("[{}]".format(tgt_lang))])]) 
            else:
                source = torch.cat([source, source.new([self.token])])
                target = torch.cat([target, target.new([self.token])])
        return source, target


import copy
class UTUTDataset(DenoisingDataset):
    def __init__(self, dataset, sizes, vocab, mask_idx, mask_whole_words, shuffle, seed, mask, mask_random, insert, rotate, permute_sentences,
            bpe, replace_length, mask_length, poisson_lambda, eos=None, item_transform_func=None, tgt_sizes=None):
        super().__init__(dataset, sizes, vocab, mask_idx, mask_whole_words, shuffle, seed, mask, mask_random, insert, rotate, permute_sentences,
            bpe, replace_length, mask_length, poisson_lambda, eos, item_transform_func)
        self.tgt_sizes = tgt_sizes

    def __getitem__(self, index):
        with data_utils.numpy_seed(self.seed, self.epoch, index):
            source, target = self.dataset[index]

            src_lang = self.dataset.src_langs[index]
            tgt_lang = self.dataset.tgt_langs[index]

            assert source[-1] == self.vocab.index("[{}]".format(src_lang))
            assert target[-1] == self.vocab.index("[{}]".format(tgt_lang))

            # if np.random.random() < 0.5:
            #     target, source = source, target
            #     tgt_lang, src_lang = src_lang, tgt_lang

            source_original = copy.deepcopy(source)

            if self.permute_sentence_ratio > 0.0:
                source = self.permute_sentences(source, self.permute_sentence_ratio)

            if self.mask_ratio > 0:
                source = self.add_whole_word_mask(source, self.mask_ratio)

            if self.insert_ratio > 0:
                source = self.add_insertion_noise(source, self.insert_ratio)

            if self.rotate_ratio > 0.0 and np.random.random() < self.rotate_ratio:
                source = self.add_rolling_noise(source)
        # there can additional changes to make:
        if self.item_transform_func is not None:
            source, target = self.item_transform_func(source, target)

        assert (source >= 0).all()
        assert (source[1:-1] >= 1).all()
        assert (source <= len(self.vocab)).all()
        assert source[0] == self.vocab.bos()
        # assert source[-1] == self.eos
        assert source[-1] == self.vocab.index("[{}]".format(src_lang))
        assert target[-1] == self.vocab.index("[{}]".format(tgt_lang))
        try:
            assert len(source) == len(source_original)
        except:
            print(source)
            print(source_original)
            import pdb
            pdb.set_trace()
        return {
            "id": index,
            "source": source,
            "source_original": source_original,
            "target": target,
        }


    def add_whole_word_mask(self, source, p):
        is_word_start = self.word_starts(source)
        num_to_mask = int(math.ceil(is_word_start.float().sum() * p))
        num_inserts = 0
        if num_to_mask == 0:
            return source

        if self.mask_span_distribution is not None:
            lengths = self.mask_span_distribution.sample(sample_shape=(num_to_mask,))

            # Make sure we have enough to mask
            cum_length = torch.cumsum(lengths, 0)
            while cum_length[-1] < num_to_mask:
                lengths = torch.cat(
                    [
                        lengths,
                        self.mask_span_distribution.sample(sample_shape=(num_to_mask,)),
                    ],
                    dim=0,
                )
                cum_length = torch.cumsum(lengths, 0)

            # Trim to masking budget
            i = 0
            while cum_length[i] < num_to_mask:
                i += 1
            lengths[i] = num_to_mask - (0 if i == 0 else cum_length[i - 1])
            num_to_mask = i + 1
            lengths = lengths[:num_to_mask]

            # Handle 0-length mask (inserts) separately
            lengths = lengths[lengths > 0]
            num_inserts = num_to_mask - lengths.size(0)
            num_to_mask -= num_inserts
            # if num_to_mask == 0:
            #     return self.add_insertion_noise(source, num_inserts / source.size(0))
            assert (lengths > 0).all()
        else:
            lengths = torch.ones((num_to_mask,)).long()
        assert is_word_start[-1] == 0
        word_starts = is_word_start.nonzero(as_tuple=False)
        indices = word_starts[
            torch.randperm(word_starts.size(0))[:num_to_mask]
        ].squeeze(1)
        mask_random = torch.FloatTensor(num_to_mask).uniform_() < self.random_ratio

        source_length = source.size(0)
        assert source_length - 1 not in indices
        to_keep = torch.ones(source_length, dtype=torch.bool)
        is_word_start[
            -1
        ] = 255  # acts as a long length, so spans don't go over the end of doc
        if self.replace_length == 0:
            to_keep[indices] = 0
        else:
            # keep index, but replace it with [MASK]
            source[indices] = self.mask_idx
            source[indices[mask_random]] = torch.randint(
                1, len(self.vocab), size=(mask_random.sum(),)
            )

        if self.mask_span_distribution is not None:
            assert len(lengths.size()) == 1
            assert lengths.size() == indices.size()
            lengths -= 1
            while indices.size(0) > 0:
                assert lengths.size() == indices.size()
                lengths -= is_word_start[indices + 1].long()
                uncompleted = lengths >= 0
                indices = indices[uncompleted] + 1
                mask_random = mask_random[uncompleted]
                lengths = lengths[uncompleted]
                if self.replace_length != -1:
                    # delete token
                    to_keep[indices] = 0
                else:
                    # keep index, but replace it with [MASK]
                    source[indices] = self.mask_idx
                    source[indices[mask_random]] = torch.randint(
                        1, len(self.vocab), size=(mask_random.sum(),)
                    )
        else:
            # A bit faster when all lengths are 1
            while indices.size(0) > 0:
                uncompleted = is_word_start[indices + 1] == 0
                indices = indices[uncompleted] + 1
                mask_random = mask_random[uncompleted]
                if self.replace_length != -1:
                    # delete token
                    to_keep[indices] = 0
                else:
                    # keep index, but replace it with [MASK]
                    source[indices] = self.mask_idx
                    source[indices[mask_random]] = torch.randint(
                        1, len(self.vocab), size=(mask_random.sum(),)
                    )

                assert source_length - 1 not in indices

        source = source[to_keep]

        # if num_inserts > 0:
        #     source = self.add_insertion_noise(source, num_inserts / source.size(0))

        return source

    def collater(self, samples, pad_to_length=None):
        """Merge a list of samples to form a mini-batch.
        Args:
            samples (List[dict]): samples to collate
        Returns:
            dict: a mini-batch of data
        """
        return collate(
            samples, self.vocab.pad(), self.eos, self.vocab, pad_to_length=pad_to_length
        )

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return max(self.sizes[index], self.tgt_sizes[index])

def collate(
    samples,
    pad_idx,
    eos_idx,
    vocab,
    left_pad_source=False,
    left_pad_target=False,
    input_feeding=True,
    pad_to_length=None,
):
    assert input_feeding
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False, pad_to_length=None):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx,
            eos_idx=None,  # use eos_idx of each sample instead of vocab.eos()
            left_pad=left_pad,
            move_eos_to_beginning=move_eos_to_beginning,
            pad_to_length=pad_to_length,
        )

    id = torch.LongTensor([s["id"] for s in samples])
    src_tokens = merge(
        "source",
        left_pad=left_pad_source,
        pad_to_length=pad_to_length["source"] if pad_to_length is not None else None,
    )
    # sort by descending source length
    src_lengths = torch.LongTensor([s["source"].numel() for s in samples])
    src_lengths, sort_order = src_lengths.sort(descending=True)
    id = id.index_select(0, sort_order)
    src_tokens = src_tokens.index_select(0, sort_order)

    source_original = merge(
        "source_original",
        left_pad=left_pad_source,
        pad_to_length=pad_to_length["source_original"] if pad_to_length is not None else None,
    )
    source_original = source_original.index_select(0, sort_order)

    prev_output_tokens = None
    target = None
    if samples[0].get("target", None) is not None:
        target = merge(
            "target",
            left_pad=left_pad_target,
            pad_to_length=pad_to_length["target"]
            if pad_to_length is not None
            else None,
        )
        target = target.index_select(0, sort_order)
        ntokens = sum(len(s["target"]) for s in samples)

        if input_feeding:
            # we create a shifted version of targets for feeding the
            # previous output token(s) into the next decoder step
            prev_output_tokens = merge(
                "target",
                left_pad=left_pad_target,
                move_eos_to_beginning=True,
                pad_to_length=pad_to_length["target"]
                if pad_to_length is not None
                else None,
            )
            prev_output_tokens = prev_output_tokens.index_select(0, sort_order)
    else:
        ntokens = sum(len(s["source"]) for s in samples)

    batch = {
        "id": id,
        "ntokens": ntokens,
        "net_input": {
            "src_tokens": src_tokens,
            "src_lengths": src_lengths,
        },
        "source_original": source_original,
        "target": target,
        "nsentences": samples[0]["source"].size(0),
        "sort_order": sort_order,
    }
    if prev_output_tokens is not None:
        batch["net_input"]["prev_output_tokens"] = prev_output_tokens

    return batch
