CUDA_VISIBLE_DEVICES=0,1,2,3 swift infer \
    --model /fanchenghao-jk/output/v9-20260304-100930/checkpoint-200 \
    --stream true \
    --infer_backend lmdeploy \
    --max_new_tokens 2048