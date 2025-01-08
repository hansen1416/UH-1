# UH-1

## Motion Retargeting

### Dependency

To establish the environment while dealing with digital human models such as SMPL and SMPL-X, we need to create a new environment
```bash
conda create -n retarget python=3.8
conda activate retarget
pip install smplx mujoco
pip install numpy==1.19.5 scipy==1.10.1 torch==2.0.0 trimesh==4.2.0
pip install pandas==1.4.4 requests==2.32.3
pip install oauthlib==3.2.2 protobuf==5.28.1 pillow==10.4.0
pip install pytorch_kinematics==0.7.2 easydict chumpy
```

Due to copyright reasons, the files below needed to be downloaded and added manually:
```
retarget/
    PoseLib/ ----> refers to https://github.com/T-K-233/PoseLib
    human_model/
        smpl/ ---> refers to [SMPL](https://smpl.is.tue.mpg.de/)
        smplx/ --> refers to [SMPLX](https://smpl-x.is.tue.mpg.de/)
    *other files and folders in the retarget section*
*other files and folders in the repo*
```

**For SMPL**, please download version 1.1.0 for Python 2.7, and rename the `basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl` as `SMPL_NEUTRAL.pkl`.
**For SMPL-X**, please download the SMPL-X 1.1 version, and keep the `.npz` files.
The file system structure is shown in the file system below
```
retarget/
    PoseLib/
    human_model/
        smpl/
            SMPL_NEUTRAL.pkl
        smplx/ 
            SMPLX_FEMALE.npz
            SMPLX_MALE.npz
            SMPLX_NEUTRAL.npz
    *other files and folders in the retarget section*
*other files and folders in the repo*
```

### Introduction & Usage

The first step to the retargeting process is to optimize for the best `betas` param of the SMPL-X model to better reshape the human model to the humanoid pose. This process is done in the `optim_betas_smplx.py`. In this file, the default pose of the humanoid robot can be set by setting the default dof pose. The range of the `betas` param can be set to avoid over deforming of the humanoid robot. The optimized `betas` param will be exported into a new file.

```bash
# make sure you are at the root folder of this project 
cd retarget
python optim_betas_smplx.py
```

In the second step, the trajectory of human body motion in SMPL-X will be retargeted into humanoid motion description (dof pose, root translation and root rotation) in `retarget_motion_smplx.py`.

```bash
# make sure you are at the root folder of this project 
cd retarget
python retarget_motion_smplx.py
```

Dealing with data in the form of SMPL models is similar with the ones in the form of SMPLX models. In this repo we provide the following code and example files
```bash
# codes
optim_betas_smplx.py
optim_betas_smpl.py
retarget_motion_smplx.py

# preprocessed betas param files
betas_param/betas_param_r5_smplx.npy
betas_param/betas_param_r5_smpl.npy
``` 

## RL Low Level Policy

### Dependency

To establish the environment with IsaacGym, we need to create a new environment

```bash
conda create -n UH-1-rl python=3.8
conda activate UH-1-rl
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1+cu117 -f https://download.pytorch.org/whl/torch_stable.html
pip install oauthlib==3.2.2 protobuf==5.28.1

# Download the Isaac Gym binaries from https://developer.nvidia.com/isaac-gym 
cd isaacgym/python && pip install -e .
cd _/rsl_rl && pip install -e .
cd _/legged_gym && pip install -e .

pip install "torch==1.13.1" "numpy==1.23.0" pydelatin==0.2.8 wandb==0.17.5 tqdm opencv-python==4.10.0.84 ipdb pyfqmr==0.2.1 flask dill==0.3.8 gdown==5.2.0 pytorch_kinematics==0.7.4 easydict==1.13
```

Due to the file size limit of Github, the training data needed for the policy can be downloaded [here](https://drive.google.com/drive/folders/1v6G6GsZZ41hg1CsUB6meU8QIDwqsDbN6?usp=sharing). Please put the data file at `motion_lib/motion_pkl/motion_data_cmu_sample.pkl`


### Usage

To train the goal-conditioned RL policy from scratch, try

```bash
# make sure you are at the root folder of this project 
cd legged_gym/legged_gym/scripts
python train.py xxx-xx-run_name --task h1_2_mimic --device cuda:0
```

To play the policy with the checkpoint we've provided, try
```bash
# make sure you are at the root folder of this project 
cd legged_gym/legged_gym/scripts
python play.py 000-00 --task h1_2_mimic --device cuda:0
```

### Misc

For additional details, this portion of the code is derived from [here](https://github.com/chengxuxin/expressive-humanoid).