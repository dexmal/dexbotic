import numpy as np
import copy


class PadState:
    """Pad the state to a fixed dimension, i.e. model action dimension, with zeros.

    This is useful when the state dimension of dataset and the model are different.
    """

    def __init__(self, ndim: int = 32, axis: int = -1):
        """Args:
        ndim: int, the dimension to pad the state. Default: 32
        axis: int, the axis to pad the state. Default: -1
        """
        self.ndim = ndim
        self.axis = axis

    def __call__(self, episode_data_dict: dict, **kwargs) -> dict:
        if "state" not in episode_data_dict:
            # warnings.warn('action is not in the episode_data_dict, skip the PadAction transform')
            return episode_data_dict

        state = episode_data_dict["state"]
        if state.shape[self.axis] < self.ndim:
            pad_width = [(0, 0) for _ in range(len(state.shape))]
            pad_width[self.axis] = (0, self.ndim - state.shape[self.axis])
            state = np.pad(state, pad_width, mode="constant", constant_values=0)
            episode_data_dict["state"] = state
        return episode_data_dict


class PadAction:
    """Pad the action to a fixed dimension, i.e. model action dimension, with zeros.

    This is useful when the action dimension of dataset and the model are different.
    """

    def __init__(self, ndim: int = 32, axis: int = -1):
        """Args:
        ndim: int, the dimension to pad the action. Default: 32
        axis: int, the axis to pad the action. Default: -1
        """
        self.ndim = ndim
        self.axis = axis

    def __call__(self, episode_data_dict: dict, **kwargs) -> dict:
        if "action" not in episode_data_dict:
            # warnings.warn('action is not in the episode_data_dict, skip the PadAction transform')
            return episode_data_dict

        action = episode_data_dict["action"]
        if action.shape[self.axis] < self.ndim:
            pad_width = [(0, 0) for _ in range(len(action.shape))]
            pad_width[self.axis] = (0, self.ndim - action.shape[self.axis])
            action = np.pad(action, pad_width, mode="constant", constant_values=0)
            episode_data_dict["action"] = action
        return episode_data_dict


class AddAction:
    """Add the action to the episode_data_dict by shifting the state.

    Will add the `action` and `abs_action` to the episode_data_dict.

    Have no effect if the `state` is not in the episode_data_dict.
    """

    def __init__(self, predict_length: int = 1):
        """Args:
        predict_length: int, the shift length of the action. Default: 1
        """
        self.predict_length = predict_length

    def __call__(self, episode_data_dict: dict, **kwargs) -> dict:
        if "state" not in episode_data_dict:
            # warnings.warn('state is not in the episode_data_dict, skip the AddAction transform')
            return episode_data_dict

        state = episode_data_dict["state"]
        # shift the state to get the action
        action = state[self.predict_length :]
        episode_data_dict["action"] = action
        episode_data_dict["abs_action"] = action
        # cut other keys to keep the same length
        for key in episode_data_dict.keys():
            if key == "meta_data":
                continue
            episode_data_dict[key] = episode_data_dict[key][: len(action)]
        return episode_data_dict


class DeltaAction:
    """Calculate the delta action from the state and action in the episode_data_dict.

       The delta action is the action - state.

       `non_delta_mask` in the meta_data is used to specify the non-delta dimension, which will be kept unchanged in delta action.
       `periodic_mask` in the meta_data specifies the dimensions representing periodic action, which will wrap in delta action.
       `periodic_range` in the meta_data specifies the range of periodic action values, used for correcting wrapping.

       Will add the `delta_action` to the episode_data_dict and replace the `action` with `delta_action`.

       Have no effect if the `action` or `state` is not in the episode_data_dict.
    """

    def __init__(self, enable=False):
        self.enable = enable

    def __call__(self, episode_data_dict: dict, **kwargs) -> dict:
        if not self.enable:
            return episode_data_dict

        if 'state' not in episode_data_dict or 'action' not in episode_data_dict:
            # warnings.warn('state or action is not in the episode_data_dict, skip the DeltaAction transform')
            return episode_data_dict

        # the action in the episode_data_dict is the absolute action
        non_delta_mask = episode_data_dict['meta_data']['non_delta_mask']
        periodic_mask = episode_data_dict['meta_data']['periodic_mask']
        periodic_range = episode_data_dict['meta_data']['periodic_range']

        state = episode_data_dict['state']
        action = episode_data_dict['action']
        # delta action is action - state
        # FIXME: should match state dim if action is trajectory
        if action.ndim == state.ndim:
            delta_action = action - state
        elif action.ndim == state.ndim + 1:
            delta_action = action - state[..., None, :]
        else:
            raise ValueError(
                f'The dim of action {action.ndim} should be equal to or one more than the dim of state {state.ndim}')


        # Apply wrap for periodic dimensions
        if periodic_mask is not None:
            for dim in periodic_mask:
                delta_action[..., dim] = np.where(
                    delta_action[..., dim] > periodic_range / 2,
                    delta_action[..., dim] - periodic_range,
                    delta_action[..., dim],
                )
                delta_action[..., dim] = np.where(
                    delta_action[..., dim] < -periodic_range / 2,
                    delta_action[..., dim] + periodic_range,
                    delta_action[..., dim],
                )

        delta_action[..., non_delta_mask] = action[..., non_delta_mask]
        episode_data_dict['delta_action'] = delta_action
        episode_data_dict['action'] = delta_action
        return episode_data_dict


class AddTrajectory:
    """Add the trajectory to the episode_data_dict by shifting the action.

       Will add the `trajectory` to the episode_data_dict and replace the `action` with `trajectory`.

       Have no effect if the `action` is not in the episode_data_dict.
    """

    def __init__(self,
                 trajectory_length: int = 10,
                 flatten: bool = True,
                 padding_mode: str = 'last',
                 padding_action: bool = False):
        """Args:
            trajectory_length: int, the length of the trajectory. Default: 10
            padding_mode: str, the padding mode for the trajectory. Default: 'last'
            padding_action: bool, whether to pad the action if the length of the action is less than the trajectory length. Default: False
        """
        self.trajectory_length = trajectory_length
        self.flatten = flatten
        self.padding_mode = padding_mode
        self.padding_action = padding_action
        assert self.padding_mode in [
            'last', 'zero'], 'only support `last` and `zero` padding mode in constructing trajectory'

    def __call__(self, episode_data_dict: dict, **kwargs) -> dict:
        if 'action' not in episode_data_dict:
            # warnings.warn('action is not in the episode_data_dict, skip the AddTrajectory transform')
            return episode_data_dict

        episode_data_dict['meta_data']['trajectory_length'] = self.trajectory_length
        non_delta_mask = episode_data_dict['meta_data']['non_delta_mask']

        action = episode_data_dict['action']  # shape: N D
        valid_trajectory_length = len(action)

        if self.padding_action:
            action = self.pad(action, self.trajectory_length, non_delta_mask)
        else:
            assert len(
                action) >= self.trajectory_length, f'the length of the action in {episode_data_dict["meta_data"]["jsonl_file"]} should be larger than the trajectory length'

        trajectory = [action]
        for i in range(1, self.trajectory_length):
            _next_action = np.copy(action[i:])
            _next_action = self.pad(_next_action, len(action), non_delta_mask)

            trajectory.append(_next_action)
        trajectory = np.stack(trajectory, axis=-1)  # shape: N D T
        # reshape to N T D than N (T * D)
        trajectory = np.transpose(trajectory, (0, 2, 1))
        if self.flatten:
            trajectory = trajectory.reshape(trajectory.shape[0], -1)
        trajectory = trajectory[:valid_trajectory_length]
        episode_data_dict['trajectory'] = trajectory
        episode_data_dict['action'] = trajectory
        return episode_data_dict

    def pad(self, action, trajectory_length, non_delta_mask):
        if len(action) >= trajectory_length:
            return action
        else:
            if self.padding_mode == 'zero':
                padding_action = np.zeros_like(action[-1])
                padding_action[non_delta_mask] = action[-1][non_delta_mask]

            else:
                padding_action = action[-1]
        action = np.concatenate([action, np.array(
            [np.copy(padding_action) for _ in range(trajectory_length - len(action))])], axis=0)
        return action


class ActionNorm:
    def __init__(
        self,
        statistic_mapping: dict = {"default": {"min": -1, "max": 1}},
        strict: bool = True,
        use_quantiles: bool = False,
    ):
        """Normalize the action to [-1, 1] by the `statistic_mapping`.

        Args:
            statistic_mapping: dict, the **per prompt** statistic mapping of the action, including 'min' and 'max'

        Note: the statistic_mapping should has a `default` key, which is the default statistic mapping for the action.
        it is also possible to have several `[dataset]` keys, which are dicts that contain the statistic mapping for
        the specific datasets. Each `dataset` key should have a `default` key, which is the default statistic mapping
        for the dataset, and several `[prompt]` key, which is a dict that contains the statistic mapping.
        """
        self.statistic_mapping = statistic_mapping
        self.strict = strict
        self.use_quantiles = use_quantiles

    def __call__(self, episode_data_dict: dict, **kwargs) -> dict:
        for key in self.statistic_mapping.keys():
            if self.strict:
                # TODO: need a better way to handle default key
                if key in ["default"]:
                    continue
                if key not in episode_data_dict:
                    raise KeyError(
                        f"{key} is not in the episode_data_dict, please check the statistic_mapping"
                    )
                else:
                    episode_data_dict[key] = self._normalize(
                        episode_data_dict[key], self.statistic_mapping[key]
                    )
            else:
                if key in episode_data_dict:
                    episode_data_dict[key] = self._normalize(
                        episode_data_dict[key], self.statistic_mapping[key]
                    )
        return episode_data_dict

    def _normalize(self, data, stats):
        if self.use_quantiles:
            return (
                (data - stats["min"]) / (stats["max"] - stats["min"] + 1e-6) * 2.0 - 1.0
            ).astype(np.float32)
        else:
            return ((data - stats["mean"]) / (stats["std"] + 1e-6)).astype(np.float32)


class ActionNormAnd2String:
    """Normalize the action to [-1, 1] and convert the action to string.

       The action will be normalized to [-1, 1] by the `statistic_mapping` and converted to string by the `vocab_size`.

       The action string will be formatted by the `string_format`.

       Will add the `action` and `answer` to the episode_data_dict. If the `answer` is already in the episode_data_dict, it will **NOT** be replaced.

       Have no effect if the `action` is not in the episode_data_dict.
    """

    def __init__(self,
                 statistic_mapping: dict = {'default': {'min': -1, 'max': 1}},
                 vocab_size: int = 255,
                 string_format: str = ' {value}',
                 add_answer: bool = True,
                 ):
        """Args:
            statistic_mapping: dict, the **per prompt** statistic mapping of the action, including 'min' and 'max'
            vocab_size: int, the vocabulary size of the action string. Default: 255
            string_format: str, the format of the action string. Default: ' {value}'
            add_answer: bool, whether to add the answer to the episode_data_dict. Default: True

        Note: the statistic_mapping should has a `default` key, which is the default statistic mapping for the action.
        it is also possible to have several `[dataset]` keys, which are dicts that contain the statistic mapping for
        the specific datasets. Each `dataset` key should have a `default` key, which is the default statistic mapping
        for the dataset, and several `[prompt]` key, which is a dict that contains the statistic mapping.

        Example:

        {
            'default': {'min': [-1, -1, -1], 'max': [1, 1, 1]},
            'dataset1': {
                'default': {'min': [-1, -1, -1], 'max': [1, 1, 1]},
                'open the door': {'min': [-0.1, -0.54, -2], 'max': [0.1, 0.54, 2]},
            },
            'dataset2': {
                'default': {'min': [-1, -1, -1], 'max': [1, 1, 1]},
            }
        }
        """
        self.vocab_size = vocab_size
        self.statistic_mapping = statistic_mapping
        self.string_format = string_format
        self.add_answer = add_answer

        assert 'default' in self.statistic_mapping, 'the default statistic mapping should be provided'

    def __call__(self, episode_data_dict: dict, **kwargs) -> dict:
        if 'action' not in episode_data_dict:
            # warnings.warn('action is not in the episode_data_dict, skip the ActionNormAnd2String transform')
            return episode_data_dict

        action = episode_data_dict['action']
        # here we assume the prompt is the same in the episode
        prompt = episode_data_dict['prompt'][0]
        dataset = episode_data_dict['meta_data']['dataset']

        if dataset not in self.statistic_mapping:
            statistic_mapping = copy.deepcopy(self.statistic_mapping['default'])
        elif prompt not in self.statistic_mapping[dataset]:
            statistic_mapping = copy.deepcopy(
                self.statistic_mapping[dataset]['default'])
        else:
            statistic_mapping = copy.deepcopy(self.statistic_mapping[dataset][prompt])

        if isinstance(statistic_mapping['min'], (int, float)):
            statistic_mapping['min'] = [statistic_mapping['min']]
            statistic_mapping['max'] = [statistic_mapping['max']]
        if len(statistic_mapping['min']) == 1:
            statistic_mapping['min'] = np.array(
                statistic_mapping['min'] * len(action[0]))
            statistic_mapping['max'] = np.array(
                statistic_mapping['max'] * len(action[0]))

        # append the statistic mapping for the trajectory
        if 'trajectory' in episode_data_dict:
            traj_length = episode_data_dict['meta_data']['trajectory_length']
            statistic_mapping['min'] = np.concatenate(
                [statistic_mapping['min'] for _ in range(traj_length)], axis=0)
            statistic_mapping['max'] = np.concatenate(
                [statistic_mapping['max'] for _ in range(traj_length)], axis=0)

        # normalize the action
        normalized_action = self._norm_action(
            action, statistic_mapping['min'], statistic_mapping['max'])
        episode_data_dict['action'] = normalized_action

        # convert the action to string
        bin_action = self._action2bin(normalized_action, self.vocab_size)
        action_str = self._bin2string(bin_action, self.string_format)

        if self.add_answer and "answer" not in episode_data_dict:
            episode_data_dict['answer'] = action_str

        return episode_data_dict

    def _norm_action(self, action, min, max) -> np.array:
        # clip the action to min and max and normalize to [-1, 1]
        min = min.reshape(1, -1)
        max = max.reshape(1, -1)
        action = np.clip(action, min, max)
        action = (action - min) / (max - min + 1e-8) * 2 - 1
        return action

    def _action2bin(self, action, vocab_size) -> np.array:
        # convert the action to binary
        action = np.round((action + 1) / 2 * (vocab_size - 1))
        action = np.clip(action, 0, vocab_size - 1)
        return action

    def _bin2string(self, action, string_format) -> list[str]:
        # convert the binary action to string
        # action: [T D] -> [T] list of str
        action_str = [''.join([string_format.format(value=int(_))
                              for _ in action[i]]) for i in range(len(action))]
        return action_str
