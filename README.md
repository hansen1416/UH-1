# Learning from Massive Human Videos for Universal Humanoid Pose Control

<p align="left">
    <a href="https://img.shields.io/badge/PRs-Welcome-red">
        <img src="https://img.shields.io/badge/PRs-Welcome-red">
    </a>
    <a href="https://img.shields.io/github/last-commit/xlang-ai/text2reward?color=green">
        <img src="https://img.shields.io/github/last-commit/xlang-ai/text2reward?color=green">
    </a>
    <br/>
</p>


Code for paper [Learning from Massive Human Videos for Universal Humanoid Pose Control]().
Please refer to our [project page](https://usc-gvl.github.io/UH-1/) for more demonstrations and up-to-date related resources. 


## Dependencies
To establish the environment, run this code in the shell:
```shell
conda create -n UH-1 python=3.8.11
conda activate UH-1
pip install git+https://github.com/openai/CLIP.git
pip install mujoco opencv-python
```



## Usage

### Preparation

Download our model checkpoints from [here](https://huggingface.co/USC-GVL/UH-1).

```bash
git lfs install
git clone https://huggingface.co/USC-GVL/UH-1
```

### Inference

- Change the `root_path` in `inference.py` to the path of the checkpoints you just downloaded.
- Change the `prompt_list` in `inference.py` to the language prompt you what the model to generate.

- Run the following commands, and the generated humanoid motion will be stored in the `output` folder.

```bash
python inference.py
```

### Visualize

- Change the `file_list` in `visualize.py` to the generated humaoid motion file names.
- Run the following commands, and the rendered video will be stored in the `output` folder.

```bash
mjpython visualize.py
```



## Citation

If you find our work helpful, please cite us:

```bibtex

```

