# CoT-RL

1. Create conda environment

    ```shell
    conda create -n cotcot-rl python=3.10
    ```

    ```shell
    conda activate cot-rl
    ```

1. Download model

    ```shell
    cd model
    git lfs install
    git clone https://huggingface.co/daryl149/llama-2-7b-chat-hf
    ```

1. Install packages

    ```shell
    pip install - r requirements.txt
    ```

1. Install trlx
    ```shell
    cd trlx-0.7.0/
    pip install -e .
    ```

1. Use accelerate launch
    ```shell
    accelerate launch --config_file config/default_config.yaml piqa.py
    ```
