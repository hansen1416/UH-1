```
sudo cp /home/hlz/anaconda3/envs/UH-1-rl/lib/libpython3.8.so.1.0 /usr/lib/
```

```
cp /usr/lib/x86_64-linux-gnu/libstdc++.so.6 /home/hlz/anaconda3/envs/UH-1-rl/lib/
```

cp env from https://github.com/leggedrobotics/rsl_rl

```
# make sure you are at the root folder of this project 
cd legged_gym/legged_gym/scripts
python play.py 000-00 --task h1_2_mimic --device cuda:0
```
