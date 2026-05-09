# please use python=3.10/3.11, cuda12.*
# sh requirements/install_all.sh
# pip install sglang -U
pip install "vllm==0.11.0" -U
pip install "lmdeploy>=0.5,<0.10.2" -U --no-deps
pip install "transformers<5.7" "trl<0.30" peft -U
pip install auto_gptq optimum bitsandbytes "gradio<5.33" -U
pip install git+https://github.com/modelscope/ms-swift.git#egg=ms-swift[all]
pip install timm "deepspeed==0.18.9" -U
pip install decord librosa icecream soundfile -U
pip install liger_kernel nvitop pre-commit math_verify py-spy wandb swanlab -U
# flash-attn: https://github.com/Dao-AILab/flash-attention/releases