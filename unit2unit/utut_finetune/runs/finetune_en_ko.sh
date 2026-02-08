!pip install --upgrade "pip<24.1"
!pip install packaging
!pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation git+https://github.com/facebookresearch/fairseq@0338cdc
!pip install --upgrade pip
!pip install fairscale inflect num2words unidecode soundfile textacy
!pip install transformers==4.41.1
!pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" git+https://github.com/NVIDIA/apex.git

filename=$(basename "$0")
languages=$(echo "$filename" | sed -E 's/.*finetune_([a-z]+_[a-z]+)\.sh/\1/')
IFS='_' read -ra langs <<< "$languages"
lang1=${langs[0]}
lang2=${langs[1]}

PYTHONPATH=fairseq \
OMP_NUM_THREADS=1 \
python finetune_${lang1}_${lang2}.py data/dataset_mbart_ft_bin_data/${lang1}/${lang2} \
--arch mbart_large \
--task translation_from_pretrained_bart \
--criterion label_smoothed_cross_entropy \
--user-dir ./
