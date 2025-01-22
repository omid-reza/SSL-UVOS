python -W ignore src/eval_oneshot.py \
        --basepath ./DAVIS \
        --batch_size 1 \
        --num_t 1 \
        --output_path test_log \
        --dataset DAVIS2017 \
        --resolution 320 480 \
        --ratio 10 \
        --tau 1.0 \
        --save_path DAVIS_Attn \
        --resume_path ./rgb_models/checkpoint_dino-s-8.pth \
        --world_size 1

#rm davis2017-evaluation/results/unsupervised/rvos/*.csv
#
#python davis2017-evaluation/evaluation_method.py \
#        --davis_path ./DAVIS \
#        --task unsupervised \
#        --results_path davis2017-evaluation/results/unsupervised/rvos
