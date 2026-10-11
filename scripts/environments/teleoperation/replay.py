"""Script to run a leisaac replay with leisaac in the simulation."""

"""Launch Isaac Sim Simulator first."""
import multiprocessing

if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", force=True)
import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="leisaac replay for leisaac in the simulation.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--sim_hz", type=int, default=None, help="Physics frequency; defaults to dataset metadata.")
parser.add_argument("--action_hz", type=int, default=None, help="Replay action frequency; defaults to recording_hz.")
parser.add_argument(
    "--dataset_file", type=str, default="./datasets/dataset.hdf5", help="File path to load recorded demos."
)
parser.add_argument(
    "--replay_mode",
    type=str,
    default="action",
    choices=["action", "state"],
    help="Replay mode, action: replay the action, state: replay the state.",
)
parser.add_argument(
    "--select_episodes",
    type=int,
    nargs="+",
    default=[],
    help="A list of episode indices to replayed. Keep empty to replay all episodes.",
)
parser.add_argument(
    "--task_type",
    type=str,
    default=None,
    help=(
        "Specify task type. If your dataset is recorded with keyboard, you should set it to 'keyboard', otherwise not"
        " to set it and keep default value None."
    ),
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

if args_cli.sim_hz is not None and args_cli.sim_hz <= 0:
    parser.error("--sim_hz must be positive")
if args_cli.action_hz is not None and args_cli.action_hz <= 0:
    parser.error("--action_hz must be positive")
if args_cli.sim_hz is not None and args_cli.action_hz is not None and args_cli.sim_hz % args_cli.action_hz != 0:
    parser.error("--sim_hz must be divisible by --action_hz")

app_launcher_args = vars(args_cli)

# launch omniverse app
app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app

import contextlib
import json
import os
import time

import gymnasium as gym
import h5py
import numpy as np
import torch
from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv
from isaaclab.sensors import CameraCfg
from isaaclab.utils.datasets import EpisodeData, HDF5DatasetFileHandler
from isaaclab_tasks.utils import parse_env_cfg
from leisaac.utils.env_utils import (
    dynamic_reset_gripper_effort_limit_sim,
    get_task_type,
)

import leisaac  # noqa: F401


def load_replay_episode(dataset_file_handler, episode_name: str, replay_mode: str) -> EpisodeData:
    """Load only initial state and the trajectory required by the replay mode."""
    episode_group = dataset_file_handler._hdf5_data_group[episode_name]
    required_fields = ("initial_state", "actions" if replay_mode == "action" else "states")

    def load_item(item):
        if isinstance(item, h5py.Group):
            return {key: load_item(value) for key, value in item.items()}
        return torch.from_numpy(np.asarray(item))

    missing = [field for field in required_fields if field not in episode_group]
    if missing:
        raise KeyError(f"Episode '{episode_name}' is missing replay fields: {missing}")

    episode = EpisodeData()
    episode.data = {field: load_item(episode_group[field]) for field in required_fields}
    episode.seed = episode_group.attrs.get("seed")
    episode.success = episode_group.attrs.get("success")
    return episode


def move_tensors_to_device(data, device):
    if isinstance(data, dict):
        return {key: move_tensors_to_device(value, device) for key, value in data.items()}
    if isinstance(data, torch.Tensor):
        return data.to(device)
    return data


class RateLimiter:
    """Convenience class for enforcing rates in loops."""

    def __init__(self, hz):
        """
        Args:
            hz (int): frequency to enforce
        """
        self.hz = hz
        self.last_time = time.time()
        self.sleep_duration = 1.0 / hz
        self.render_period = min(0.0166, self.sleep_duration)

    def sleep(self, env):
        """Attempt to sleep at the specified rate in hz."""
        next_wakeup_time = self.last_time + self.sleep_duration
        while time.time() < next_wakeup_time:
            time.sleep(self.render_period)
            env.sim.render()

        self.last_time = self.last_time + self.sleep_duration

        # detect time jumping forwards (e.g. loop is too slow)
        if self.last_time < time.time():
            while self.last_time < time.time():
                self.last_time += self.sleep_duration


def get_next_action(episode_data: EpisodeData, return_state: bool = False, task_type: str = None):
    if return_state:
        next_state = episode_data.get_next_state()
        if next_state is None:
            return None
        if task_type == "bi-so101leader":
            left_joint_pos = next_state["articulation"]["left_arm"]["joint_position"]
            right_joint_pos = next_state["articulation"]["right_arm"]["joint_position"]
            return torch.cat([left_joint_pos, right_joint_pos], dim=0)
        else:
            return next_state["articulation"]["robot"]["joint_position"]
    else:
        return episode_data.get_next_action()


def main():
    """Replay episodes loaded from a file."""

    # Load dataset
    if not os.path.exists(args_cli.dataset_file):
        raise FileNotFoundError(f"The dataset file {args_cli.dataset_file} does not exist.")
    dataset_file_handler = HDF5DatasetFileHandler()
    dataset_file_handler.open(args_cli.dataset_file)
    episode_count = dataset_file_handler.get_num_episodes()

    env_args = json.loads(dataset_file_handler._hdf5_data_group.attrs.get("env_args", "{}"))
    frequency_metadata = env_args.get("frequency", {})
    sim_hz = args_cli.sim_hz or frequency_metadata.get("simulation_hz", 60)
    action_hz = args_cli.action_hz or frequency_metadata.get(
        "recording_hz", frequency_metadata.get("action_hz", 60)
    )
    if sim_hz < action_hz or sim_hz % action_hz != 0:
        raise ValueError(f"sim_hz ({sim_hz}) must be >= and divisible by action_hz ({action_hz})")

    if episode_count == 0:
        print("No episodes found in the dataset.")
        exit()

    episode_indices_to_replay = args_cli.select_episodes
    if len(episode_indices_to_replay) == 0:
        episode_indices_to_replay = list(range(episode_count))

    num_envs = args_cli.num_envs

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=num_envs)
    task_type = get_task_type(args_cli.task, args_cli.task_type)
    env_cfg.use_teleop_device(task_type)
    env_cfg.sim.dt = 1.0 / sim_hz
    env_cfg.decimation = sim_hz // action_hz
    env_cfg.sim.render_interval = sim_hz // action_hz
    for value in vars(env_cfg.scene).values():
        if isinstance(value, CameraCfg):
            value.update_period = 1.0 / action_hz

    # Disable all recorders and terminations
    is_direct_env = "Direct" in args_cli.task
    if is_direct_env:
        env_cfg.never_time_out = True
        env_cfg.manual_terminate = True
        env_cfg.recorders = {}
    else:
        env_cfg.recorders = {}
        env_cfg.terminations = {}

    # create environment from loaded config
    env: ManagerBasedRLEnv | DirectRLEnv = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    # Get idle action (idle actions are applied to envs without next action)
    if hasattr(env_cfg, "idle_action"):
        idle_action = env_cfg.idle_action.repeat(num_envs, 1).to(env.device)
    else:
        idle_action = torch.zeros(env.action_space.shape, device=env.device)

    # reset before starting
    if hasattr(env, "initialize"):
        env.initialize()
    env.reset()

    print(f"[Replay Frequency] simulation={sim_hz} Hz, action={action_hz} Hz")
    rate_limiter = RateLimiter(action_hz)

    # simulate environment -- run everything in inference mode
    episode_names = list(dataset_file_handler.get_episode_names())
    replayed_episode_count = 0
    with contextlib.suppress(KeyboardInterrupt) and torch.inference_mode():
        while simulation_app.is_running() and not simulation_app.is_exiting():
            env_episode_data_map = {index: EpisodeData() for index in range(num_envs)}
            has_next_action = True
            while has_next_action:
                # initialize actions with idle action so those without next action will not move
                actions = idle_action.clone()
                has_next_action = False
                for env_id in range(num_envs):
                    env_next_action = get_next_action(
                        env_episode_data_map[env_id], return_state=args_cli.replay_mode == "state", task_type=task_type
                    )
                    if env_next_action is None:
                        next_episode_index = None
                        while episode_indices_to_replay:
                            next_episode_index = episode_indices_to_replay.pop(0)
                            if next_episode_index < episode_count:
                                break
                            next_episode_index = None

                        if next_episode_index is not None:
                            replayed_episode_count += 1
                            print(f"{replayed_episode_count :4}: Loading #{next_episode_index} episode to env_{env_id}")
                            episode_data = load_replay_episode(
                                dataset_file_handler,
                                episode_names[next_episode_index],
                                args_cli.replay_mode,
                            )
                            env_episode_data_map[env_id] = episode_data
                            # Set initial state for the new episode
                            initial_state = move_tensors_to_device(episode_data.get_initial_state(), env.device)
                            env.reset_to(
                                initial_state,
                                torch.tensor([env_id], device=env.device),
                                seed=int(episode_data.seed) if episode_data.seed is not None else None,
                                is_relative=True,
                            )
                            # Get the first action for the new episode
                            env_next_action = get_next_action(
                                env_episode_data_map[env_id],
                                return_state=args_cli.replay_mode == "state",
                                task_type=task_type,
                            )
                            has_next_action = True
                        else:
                            continue
                    else:
                        has_next_action = True
                    actions[env_id] = env_next_action.to(env.device)
                if args_cli.replay_mode == "action":
                    if env.cfg.dynamic_reset_gripper_effort_limit:
                        dynamic_reset_gripper_effort_limit_sim(env, task_type)
                env.step(actions)
                rate_limiter.sleep(env)
            break
    # Close environment after replay in complete
    plural_trailing_s = "s" if replayed_episode_count > 1 else ""
    print(f"Finished replaying {replayed_episode_count} episode{plural_trailing_s}.")
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
