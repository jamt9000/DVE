ipy cache_feats.py -- --config configs/cache_feats_smallnet_mafl_64d_dve_128in_keypoints-ep57.json
%run -i train.py --config configs/cached_smallnet_mafl_64d_dve_128in_keypoints-ep57-v1.json
python train.py --config configs/cached_smallnet_mafl_64d_dve_128in_keypoints-ep57-v1.json --disable_workers 0 --profile 1 --disable_vis 1
torch.cuda.empty_cache()
python train.py --config configs/smallnet_mafl_64d_dve_128in_keypoints-ep57-v1-noup.json --disable_workers 0 --profile 0 --vis 0

python3 train.py --config configs/smallnet_aflw_64d_dve_128in_keypoints-ep57-v1.json
