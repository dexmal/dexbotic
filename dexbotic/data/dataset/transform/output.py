import math
import numpy as np


class Relative2Delta:
    def __call__(self, episode_data_dict: dict, **kwargs) -> dict:
        if "action" not in episode_data_dict:
            # warnings.warn('state or action is not in the episode_data_dict, skip the AbsoluteAction transform')
            return episode_data_dict

        non_delta_mask = episode_data_dict["meta_data"].get("non_delta_mask", [-1])
        periodic_mask = episode_data_dict["meta_data"].get("periodic_mask", None)
        periodic_range = episode_data_dict["meta_data"].get("periodic_range", math.pi)

        action = episode_data_dict["action"]

        delta_action = action.copy()
        # FIXME: require more flexible masking when gripper dim is not matched
        # between action and state
        # abs_action[..., :non_delta_mask] = (
        #     state[:, None, :non_delta_mask] + action[..., :non_delta_mask]
        # )
        if action.ndim == 2:
            delta_action[1:, :] = action[1:, :] - action[:-1, :]
        elif action.ndim == 3:
            delta_action[:, 1:, :] = action[:, 1:, :] - action[:, :-1, :]
        else:
            raise ValueError(
                f'The dim of action {action.ndim} should be 2 or 3'
            )

        delta_action[..., non_delta_mask] = action[..., non_delta_mask]

        # Apply correction for rotation dimensions
        if periodic_mask is not None:
            for dim in periodic_mask:
                delta_action[..., dim] = np.where(
                    delta_action[..., dim] > periodic_range,
                    delta_action[..., dim] - periodic_range * 2,
                    delta_action[..., dim],
                )
                delta_action[..., dim] = np.where(
                    delta_action[..., dim] < -periodic_range,
                    delta_action[..., dim] + periodic_range * 2,
                    delta_action[..., dim],
                )

        episode_data_dict["delta_action"] = delta_action
        episode_data_dict["action"] = delta_action
        return episode_data_dict


class AbsoluteAction:
    def __call__(self, episode_data_dict: dict, **kwargs) -> dict:
        if "state" not in episode_data_dict or "action" not in episode_data_dict:
            # warnings.warn('state or action is not in the episode_data_dict, skip the AbsoluteAction transform')
            return episode_data_dict

        non_delta_mask = episode_data_dict["meta_data"].get("non_delta_mask", [-1])
        periodic_mask = episode_data_dict["meta_data"].get("periodic_mask", None)
        periodic_range = episode_data_dict["meta_data"].get("periodic_range", math.pi)

        state = episode_data_dict["state"]
        action = episode_data_dict["action"]

        abs_action = action.copy()
        # FIXME: require more flexible masking when gripper dim is not matched
        # between action and state
        # abs_action[..., :non_delta_mask] = (
        #     state[:, None, :non_delta_mask] + action[..., :non_delta_mask]
        # )
        if action.ndim == state.ndim:
            abs_action = state + action
        elif action.ndim == state.ndim + 1:
            abs_action = state[..., None, :] + action
        else:
            raise ValueError(
                f'The dim of action {action.ndim} should be equal to or one more than the dim of state {state.ndim}'
            )

        abs_action[..., non_delta_mask] = action[..., non_delta_mask]

        # Apply correction for rotation dimensions
        if periodic_mask is not None:
            for dim in periodic_mask:
                abs_action[..., dim] = np.where(
                    abs_action[..., dim] > periodic_range,
                    abs_action[..., dim] - periodic_range * 2,
                    abs_action[..., dim],
                )
                abs_action[..., dim] = np.where(
                    abs_action[..., dim] < -periodic_range,
                    abs_action[..., dim] + periodic_range * 2,
                    abs_action[..., dim],
                )

        episode_data_dict["abs_action"] = abs_action
        episode_data_dict["action"] = abs_action
        return episode_data_dict


class ActionDenorm:
    def __init__(
        self,
        statistic_mapping: dict = {"default": {"min": -1, "max": 1}},
        strict: bool = True,
        use_quantiles: bool = False,
    ):
        """Denormalize the action from [-1, 1] to the original range by the `statistic_mapping`.

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
                if key not in episode_data_dict:
                    raise KeyError(
                        f"{key} is not in the episode_data_dict, please check the statistic_mapping"
                    )
                else:
                    episode_data_dict[key] = self._denormalize(
                        episode_data_dict[key], self.statistic_mapping[key]
                    )
            else:
                if key in episode_data_dict:
                    episode_data_dict[key] = self._denormalize(
                        episode_data_dict[key], self.statistic_mapping[key]
                    )
        return episode_data_dict

    def _denormalize(self, data, stats):
        if self.use_quantiles:
            if stats["max"].shape[-1] != data.shape[-1]:
                stats["max"] = np.concatenate(
                    [stats["max"], np.ones(data.shape[-1] - stats["max"].shape[-1])],
                    axis=-1,
                )
                stats["min"] = np.concatenate(
                    [stats["min"], -np.ones(data.shape[-1] - stats["min"].shape[-1])],
                    axis=-1,
                )
            return (data + 1) / 2 * (stats["max"] - stats["min"] + 1e-6) + stats["min"]
        else:
            if stats["mean"].shape[-1] != data.shape[-1]:
                stats["mean"] = np.concatenate(
                    [stats["mean"], np.zeros(data.shape[-1] - stats["mean"].shape[-1])],
                    axis=-1,
                )
                stats["std"] = np.concatenate(
                    [stats["std"], np.ones(data.shape[-1] - stats["std"].shape[-1])],
                    axis=-1,
                )
            return data * (stats["std"] + 1e-6) + stats["mean"]
