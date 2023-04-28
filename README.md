## Code Structure 

In the script folder, you can reimplement all the  experiment charts, results. You can run `xxx.sh` to run the experiment directly.

- `algorithms`: contain main algorithm and its variants
  - `sac.py`  SAC as base RL algorithm
  - `rad.py`  RAD as base RL algorithm
  - `rad_byol_dema` our PID implementation. We add a target line for dyanmics model for perspective-invariance loss.

