# Vimeo-90K VSR Training

The local Kaggle mirror is stored as raw Vimeo-90K septuplets:

```text
data/raw/vimeo90k_kaggle/
  readme.txt
  sep_trainlist.txt
  sep_testlist.txt
  sequence/
    00081/
      0001/
        im1.png
        im2.png
        ...
        im7.png
```

The training path reads these frames directly. It does not rewrite, resize,
copy, or symlink the downloaded frames. LR inputs are generated in memory from
HR crops for each batch.

The training environment needs the same core packages as the existing SISR
pipeline: PyTorch, NumPy, OpenCV, PyYAML, matplotlib, and tqdm.

The Vimeo configs set `run_group: "vsr"`, so new experiments are written under:

```text
runs/vsr/<model_name>/Vimeo90K/exp_...
```

Run the small temporal baseline first:

```bash
python3 train.py --config configs/train_vimeo90k_video_espcn_x2_3f.yaml
```

Then run the recurrent VSR models:

```bash
python3 train.py --config configs/train_vimeo90k_vsrbasic_x2_5f.yaml
python3 train.py --config configs/train_vimeo90k_vsrplusplus_x2_7f.yaml
```

Or run all three experiments sequentially:

```bash
python3 run_experiments.py configs/sweep_vimeo90k_vsr_x2.yaml
```

The three configs use:

```text
VideoESPCN    3 LR frames -> HR center frame
VSRBasic      5 LR frames -> HR center frame
VSRPlusPlus   7 LR frames -> HR center frame
```

`VSRPlusPlus` is a lightweight BasicVSR++-style model for this repo. It uses
second-order bidirectional propagation and temporal attention, but avoids
optical-flow and deformable-convolution dependencies so the PC experiment can
run from the existing training stack.
