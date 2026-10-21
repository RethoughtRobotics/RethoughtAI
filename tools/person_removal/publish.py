"""Step 3 of the pipeline: push a processed dataset to the Hugging Face Hub and check every episode decodes.

Needs huggingface_hub and, for the check, lerobot (any environment that trains on the dataset). The check
loads the pushed revision and reads the first and last frame of every episode on every camera.
"""

from pathlib import Path


def push(folder: Path, repo_id: str, private: bool = True) -> str:
    """Upload `folder` as dataset `repo_id`, tagged v3.0 as LeRobot expects. Returns the commit SHA."""
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, repo_type='dataset', private=private, exist_ok=True)
    api.upload_large_folder(repo_id=repo_id, folder_path=str(folder), repo_type='dataset', private=private)
    api.create_tag(repo_id, tag='v3.0', repo_type='dataset', exist_ok=True)
    sha = api.dataset_info(repo_id).sha
    print(f'pushed https://huggingface.co/datasets/{repo_id} at {sha}')
    return sha


def check(repo_id: str, revision: str):
    """Decode the first and last frame of every episode of the pushed dataset."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(repo_id, revision=revision)
    episodes = ds.meta.episodes
    for i in range(len(episodes)):
        for idx in (int(episodes[i]['dataset_from_index']), int(episodes[i]['dataset_to_index']) - 1):
            item = ds[idx]
            assert int(item['episode_index']) == int(episodes[i]['episode_index']), (i, idx)
    print(f'{repo_id}@{revision}: first and last frame of all {len(episodes)} episodes decode '
          f'on {len(ds.meta.video_keys)} cameras')  # fmt: skip
