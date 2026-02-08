for f in /path/to/datasets/aihub_a2a_unit_dedup/test/ko/*.txt; do
    cat "$f"
    echo ""
done > /path/to/datasets/aihub_a2a_unit_dedup/raw_data/test.ko

for f in /path/to/datasets/aihub_a2a_unit_dedup/test/en/*.txt; do
    cat "$f"
    echo ""
done > /path/to/datasets/aihub_a2a_unit_dedup/raw_data/test.en



fairseq-preprocess \
    --source-lang en \
    --target-lang ko \
    --trainpref /home/2022113135/datasets/aihub_a2a_unit_dedup/raw_data/train \
    --validpref /home/2022113135/datasets/aihub_a2a_unit_dedup/raw_data/valid \
    --testpref /home/2022113135/datasets/aihub_a2a_unit_dedup/raw_data/test \
    --destdir /home/2022113135/jjs/av2av/unit2unit/utut_finetune/data/dataset_mbart_ft_bin_data/en/aihub_ko \
    --srcdict /home/2022113135/jjs/av2av/unit2unit/dict.txt \
    --tgtdict /home/2022113135/jjs/av2av/unit2unit/dict.txt \
    --workers 4
# pretrained model의 dict tkdyd

# 시작 위치 : av2av/unit2unit/utut_finetune/
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=/home/2022113135/jjs/av2av/fairseq OMP_NUM_THREADS=1 python finetune_en_ko.py data/dataset_mbart_ft_bin_data/en/ko --arch mbart_large \
--task translation_from_pretrained_bart \
--criterion label_smoothed_cross_entropy \
# --tensorboard_logdir ./utut_ckpt/unit_mbart_multilingual_ft/en_ko \
--user-dir ./