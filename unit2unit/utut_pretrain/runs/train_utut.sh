CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 \
PYTHONPATH=fairseq \
python train_utut.py ./dataset \
--arch mbart_large \
--task utut_pretraining \
--criterion label_smoothed_cross_entropy \
--user-dir ./
