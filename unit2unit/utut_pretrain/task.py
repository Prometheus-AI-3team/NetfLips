# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import torch
import numpy as np
from omegaconf import II

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
from fairseq.data.encoders.utils import get_whole_word_mask
from fairseq.tasks import register_task

from fairseq.tasks.denoising import DenoisingConfig, DenoisingTask
from fairseq.tasks.multilingual_denoising import MultilingualDenoisingConfig, MultilingualDenoisingTask

from .dataset import UTUTDataset, UnitToUnitDatasetCreator, MyPrependTokenDataset, MyAppendTokenDataset, STUTDataset

logger = logging.getLogger(__name__)


@register_task("utut_pretraining", dataclass=MultilingualDenoisingConfig)
class UTUTTask(MultilingualDenoisingTask):

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = self.cfg.data.split(":")
        assert len(paths) > 0
        data_path = paths[(epoch - 1) % len(paths)]
        # split_path = os.path.join(data_path, split)

        if self.cfg.langs is None:
            languages = sorted(
                [
                    name
                    for name in os.listdir(data_path)
                    if os.path.isdir(os.path.join(data_path, name))
                ]
            )
        else:
            languages = self.cfg.langs.split(",")
        #     for name in languages:
        #         p = os.path.join(data_path, name)
        #         assert os.path.exists(p), "data not found: {}".format(p)

        logger.info("Training on {0} languages: {1}".format(len(languages), languages))
        logger.info(
            "Language to id mapping: ", {lang: id for id, lang in enumerate(languages)}
        )

        # mask_whole_words = get_whole_word_mask(self.cfg.bpe, self.dictionary)
        # language_without_segmentations = self.cfg.no_whole_word_mask_langs.split(",")
        # lang_datasets = []

        is_train_split = split.startswith("train")
        dataset = UnitToUnitDatasetCreator.from_tsv(
            root=self.cfg.data,
            split=split,
            is_train_split=is_train_split,
            epoch=epoch,
            seed=self.cfg.seed,
            src_dict=self.dictionary,
            tgt_dict=self.dictionary,
            max_target_positions=self.cfg.max_target_positions
        )

        for src_lang in set(dataset.src_langs):
            assert self.dictionary.index("[{}]".format(src_lang)) != self.dictionary.unk(), f"source language {src_lang} not in dictionary!"
        for tgt_lang in set(dataset.tgt_langs):
            assert self.dictionary.index("[{}]".format(tgt_lang)) != self.dictionary.unk(), f"target language {tgt_lang} not in dictionary!"

        dataset = MyPrependTokenDataset(dataset, self.source_dictionary.bos())
        dataset = MyAppendTokenDataset(dataset, self.source_dictionary.eos(), replace_with_lang_token=True, dictionary=self.dictionary)

        lang_mask_whole_words = None #(
        #     mask_whole_words
        #     if language not in language_without_segmentations
        #     else None
        # )
        dataset = UTUTDataset(
            dataset,
            dataset.sizes,
            self.dictionary,
            self.mask_idx,
            lang_mask_whole_words,
            shuffle=self.cfg.shuffle_instance,
            seed=self.cfg.seed,
            mask=self.cfg.mask if is_train_split else 0.0,
            mask_random=self.cfg.mask_random,
            insert=self.cfg.insert,
            rotate=self.cfg.rotate,
            permute_sentences=self.cfg.permute_sentences,
            bpe=self.cfg.bpe,
            replace_length=self.cfg.replace_length,
            mask_length=self.cfg.mask_length,
            poisson_lambda=self.cfg.poisson_lambda,
            eos=None,
            # if not self.cfg.add_lang_token
            # else self.source_dictionary.index("[{}]".format(language)),
            tgt_sizes=dataset.tgt_sizes
        )

        with data_utils.numpy_seed(self.cfg.seed + epoch):
            shuffle = np.random.permutation(len(dataset))

        self.datasets[split] = SortDataset(
            dataset,
            sort_order=[
                shuffle,
                dataset.sizes,
            ],
        )

    def build_generator(
        self,
        models,
        args,
        seq_gen_cls=None,
        extra_gen_cls_kwargs=None,
    ):
        # if self.data_cfg.prepend_tgt_lang_tag and args.prefix_size != 1:
        #     raise ValueError(
        #         'Please set "--prefix-size 1" since '
        #         "target language ID token is prepended as BOS."
        #     )
        # lang_token_ids = {
        #     i
        #     for s, i in self.tgt_dict.indices.items()
        #     if SpeechToTextDataset.is_lang_tag(s)
        # }

        languages = self.cfg.langs.split(",")

        lang_token_ids = {
            self.dictionary.index("[{}]".format(language))
            for language in languages
        }

        if extra_gen_cls_kwargs is None:
            extra_gen_cls_kwargs = {}
        extra_gen_cls_kwargs["symbols_to_strip_from_output"] = lang_token_ids

        # eos_token = (
        #     args.eos_token
        #     if "eos_token" in args and args.eos_token is not None
        #     else self.data_cfg.config.get("eos_token", None)
        # )

        # if self.data_cfg.prepend_bos_and_append_tgt_lang_tag and not eos_token:
        #     raise Warning(
        #         "Please provide --eos_token to replace eos in sequence generator"
        #     )

        target_language = self.target_language

        # eos_id = self.tgt_dict.index(eos_token) if eos_token else None
        # extra_gen_cls_kwargs["eos"] = eos_id
        extra_gen_cls_kwargs["eos"] = self.dictionary.index("[{}]".format(target_language))

        extra_gen_cls_kwargs["tokens_to_suppress"] = \
            ["[{}]".format(language) for language in languages if language not in [target_language]] \
            + [self.dictionary[self.mask_idx]]

        # from .sequence_generator import UTUTSequenceGenerator
        return super().build_generator(
            models,
            args,
            seq_gen_cls=seq_gen_cls, #UTUTSequenceGenerator,
            extra_gen_cls_kwargs=extra_gen_cls_kwargs,
        )
