"""
Visualize experiment data in fiftyone

Usage:
  python visualize_fiftyone.py fly.yaml

TODO: error cannot use label_strong value to find frames.
"""
import yaml
from pathlib import Path
from omegaconf import OmegaConf
import os
from action.data.datamodule import all_datasets
import numpy as np
from pytorch_lightning import seed_everything
import fiftyone as fo
from tqdm import tqdm
BASE_DIR = Path(__file__).resolve().parent.parent.parent
from fire import Fire
toy_data = {
  "ibl": str(BASE_DIR / "data/ibl"),
  "fly": str(BASE_DIR / "data/fly"),
}

TOY_CONFIG = {
  "fly": str(BASE_DIR / "scripts/script_configs/fly.yaml"),
  "fly_daart": str(BASE_DIR / "scripts/script_configs/fly_daart.yaml"),
  "ibl": str(BASE_DIR / "scripts/script_configs/ibl_paw.yaml"),
}

def load_cfg_demo(data) -> dict:
  """Load all toy data config file without hydra."""
  cfg = yaml.load(open(str(TOY_CONFIG[data])), Loader=yaml.FullLoader)
  return OmegaConf.create(cfg)

def load_cfg(data) -> dict:
  """Load all toy data config file without hydra."""
  cfg = yaml.load(open(str(data)), Loader=yaml.FullLoader)
  return OmegaConf.create(cfg)


def visualize_config(config_file):
  if not os.path.isabs(config_file):
    config_file = str(BASE_DIR / "scripts/script_configs/{}".format(config_file))

  args = load_cfg(config_file)
  if args.seed is not None:
    seed_everything(args.seed)

  # get data_cfg
  cfg = args.data_cfg
  cfg.num_workers = os.cpu_count()
  #cfg.input_type = input_type
  cfg.normalize_markers = False  # do not normalize for visualization
  cfg.as_numpy = True  # do not send to gpu for visualization
  # visualize from loader:
  extra_kwargs = {
    "sequence_pad": 0,  # do not pad for visualization
    "lambda_weak": args.module_cfg.lambda_weak,
    "lambda_strong": args.module_cfg.lambda_strong,
    "lambda_pred": 0,
  }

  ind_data = all_datasets[cfg.test_set](cfg, **extra_kwargs)
  ind_data.setup(stage="predict")
  cumulative_sizes = ind_data.dataset.cumulative_sizes
  cumulative_sizes = np.asarray([0] + cumulative_sizes)
  print('Building fiftyone dataset to visualize labeled data')
  fo_dataset = fo.Dataset(name=args.project_name)

  video_idx = 0
  for batch_idx, batch in enumerate(tqdm(ind_data.dataset)):
    if batch_idx == cumulative_sizes[video_idx]:
      video = os.path.join(cfg.data_dir, 'videos', cfg.expt_ids[video_idx] + ".mp4")
      if os.path.exists(video):
        print("video exists", flush=True)
        video_sample = fo.Sample(filepath=video)
        video_sample.compute_metadata()
        img_width = video_sample.metadata['frame_width']
        img_height = video_sample.metadata['frame_height']
      else:
        #TODO: visualize without video:
        raise NotImplementedError
      video_idx += 1

    # Convert the PyTorch sample with traces to FiftyOne format:
    sequences_indices = batch['sequence_idx']#.detach().numpy()
    # for each sample, for each frame
    for sample_idx, sample_sequence in enumerate(sequences_indices):
      for frame_idx, frame_number in enumerate(sample_sequence):
        frame_number = int(frame_number)
        if 'markers' in batch.keys():
          # TODO: embeddings should not be normalized
          frame_points = batch['markers'][sample_idx][frame_idx].reshape((2, -1)).T
          new_points = [(kp[0]/img_width, kp[1]/img_height,) for kp in frame_points]
          keypoint = fo.Keypoint(points=new_points)
          kps = fo.Keypoints(keypoints=[keypoint])
          video_sample.frames[frame_number + 1]["markers"] = kps
        # other keys:
        if 'labels_strong' in batch.keys():
          label_strong = fo.Classification(label=str(batch['labels_strong'][sample_idx][frame_idx]))
          video_sample.frames[frame_number + 1]["labels_strong"] = label_strong
        if 'labels_weak' in batch.keys():
          label_weak = fo.Classification(label=str(batch['labels_weak'][sample_idx][frame_idx]))
          video_sample.frames[frame_number + 1]["labels_weak"] = label_weak
    # add video to data before next batch is added to video:
    #fo_dataset.add_sample(video_sample)
    #break
    if (batch_idx + 1) == cumulative_sizes[video_idx]:
      print('Add fields from video sample {}'.format(cfg.expt_ids[video_idx-1]), flush=True)
      fo_dataset.add_sample(video_sample)
    #"""
  print(fo_dataset)
  print('Launch fityone app', flush=True)
  # Launch the FiftyOne App to visualize your dataset
  session = fo.launch_app(dataset=fo_dataset)
  # block execution until app is closed
  session.wait()

if __name__ == '__main__':
    Fire(visualize_config)