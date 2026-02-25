import os
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F
from loguru import logger

from dexbotic.exp.base_exp import TrainerConfig
from dexbotic.exp.rl.rl_base import GRPOConfig, GRPOTrainer, RLEnvironmentConfig
from dexbotic.exp.trainer import DexboticTrainer

try:
    from flash_attn.ops.triton.cross_entropy import cross_entropy_loss

    FLAH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE = True
except ImportError:
    FLAH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE = False


def gather_from_labels(data, label):
    output = torch.gather(data, -1, label.unsqueeze(-1)).squeeze(-1)
    return output


def logprobs_from_logits(logits, labels):
    if FLAH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE:
        batch_dim = logits.shape[:-1]
        last_dim = logits.shape[-1]
        logits = logits.reshape(-1, last_dim)
        labels = labels.reshape(-1)
        output = logprobs_from_logits_flash_attn(logits, labels)
        output = output.view(*batch_dim)
    else:
        output = logprobs_from_logits_naive(logits, labels)
    return output


def logprobs_from_logits_flash_attn(logits, labels):
    output = -cross_entropy_loss(logits, labels)[0]
    return output


def logprobs_from_logits_naive(logits, labels):
    logp = F.log_softmax(logits, dim=-1)
    logpy = gather_from_labels(logp, labels)
    return logpy


def entropy_from_logits(logits: torch.Tensor):
    pd = torch.nn.functional.softmax(logits, dim=-1)
    entropy = torch.logsumexp(logits, dim=-1) - torch.sum(pd * logits, dim=-1)
    return entropy


def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    eos_mask: torch.Tensor,
    index: torch.Tensor,
    epsilon: float = 1e-6,
):
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
        scores = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask

    return scores, scores


def compute_advantage(batch, gamma, lam, adv_estimator, config):
    responses = batch["responses"]
    response_length = responses.size(1) * responses.size(2)
    finish_step = batch["finish_step"] * config.model_config.action_dim
    steps = torch.arange(response_length, device=batch["responses"].device)
    steps_expanded = steps.unsqueeze(0).expand(batch["responses"].size(0), -1)
    response_mask = steps_expanded < finish_step.unsqueeze(1)

    token_level_rewards = (
        batch["token_level_rewards"]
        if "token_level_rewards" in list(batch.keys())
        else batch["token_level_scores"]
    )

    if adv_estimator == "grpo":
        token_level_rewards = batch["token_level_rewards"]
        index = batch["uid"]
        responses = batch["responses"]
        response_length = responses.size(1) * responses.size(2)
        finish_step = batch["finish_step"] * config.model_config.action_dim
        steps = torch.arange(response_length, device=batch["responses"].device)
        steps_expanded = steps.unsqueeze(0).expand(batch["responses"].size(0), -1)
        response_mask = steps_expanded < finish_step.unsqueeze(1)
        advantages, returns = compute_grpo_outcome_advantage(
            token_level_rewards=token_level_rewards, eos_mask=response_mask, index=index
        )
        batch["advantages"] = advantages
        batch["returns"] = returns
    else:
        raise NotImplementedError
    return batch


def masked_mean(values, mask, axis=None):
    mask_sum = mask.sum(axis=axis)
    masked_sum = (values * mask).sum(axis=axis)
    result = masked_sum / torch.clamp(mask_sum, min=1e-8)
    result = torch.where(mask_sum > 0, result, masked_sum * 0.0)
    return result


def kl_penalty(
    logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty
) -> torch.FloatTensor:
    if kl_penalty == "kl":
        return logprob - ref_logprob
    if kl_penalty == "abs":
        return (logprob - ref_logprob).abs()
    if kl_penalty == "mse":
        return 0.5 * (logprob - ref_logprob).square()
    if kl_penalty == "full":
        raise NotImplementedError
    raise NotImplementedError


def apply_kl_penalty(
    batch, kl_ctrl, kl_penalty="kl", action_token_len=7, action_chunks_len=8
):
    responses = batch["responses"]
    traj_length = responses.size(1) * action_chunks_len
    action_length = action_token_len
    token_level_scores = batch["token_level_scores"]
    batch_size = batch["responses"].shape[0]
    finish_step = batch["finish_step"] * action_length

    steps = torch.arange(traj_length * action_length, device=batch["responses"].device)
    steps_expanded = steps.unsqueeze(0).expand(batch["responses"].size(0), -1)
    response_mask = steps_expanded < finish_step.unsqueeze(1)

    if "ref_log_prob" in batch.keys():
        kld = kl_penalty(
            batch["old_log_probs"], batch["ref_log_prob"], kl_penalty=kl_penalty
        )
        kld = kld * response_mask
        beta = kl_ctrl.value
    else:
        beta = 0
        kld = torch.zeros_like(response_mask, dtype=torch.float32)

    token_level_rewards = token_level_scores - beta * kld
    current_kl = masked_mean(kld, mask=response_mask, axis=-1)
    current_kl = torch.mean(current_kl, dim=0).item()
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    batch["token_level_rewards"] = token_level_rewards
    metrics = {"critic/kl": current_kl, "critic/kl_coeff": beta}
    return batch, metrics


class FixedKLController:
    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current_kl, n_steps):
        pass


class RobRewardManager:
    def __init__(self, num_examine, config) -> None:
        self.num_examine = num_examine
        self.config = config

    def verify(self, data):
        completes = data["complete"].tolist()
        batch_size = data["responses"].size(0)
        assert len(completes) == batch_size
        score = [float(item) for item in completes]
        format = [1.0 for _ in range(len(completes))]

        data["acc"] = torch.tensor(
            score, dtype=torch.float32, device=data["responses"].device
        )
        data["format_correctness"] = torch.tensor(
            format, dtype=torch.float32, device=data["responses"].device
        )

        reward_metrics = {}
        format_metrics = {}
        reward_format_metrics = {}
        reward_metrics["all"] = data["acc"].mean().item()
        format_metrics["all"] = data["format_correctness"].mean().item()
        reward_format_metrics["all"] = data["acc"].mean().item()
        return score, reward_metrics, format_metrics, reward_format_metrics

    def __call__(self, data):
        reward_tensor_dict = {}
        reward_metrics = {}
        reward_tensor = torch.zeros_like(data["responses"], dtype=torch.float32)
        verifier_reward = torch.zeros_like(data["responses"], dtype=torch.float32)
        reward_tensor = reward_tensor.reshape((reward_tensor.shape[0], -1))
        verifier_reward = verifier_reward.reshape((verifier_reward.shape[0], -1))

        valid_response_length = data["finish_step"] * self.config.model.action_dim

        if "acc" in data:
            verifier_score = data["acc"].cpu().numpy().tolist()
        else:
            (
                verifier_score,
                verifier_metrics,
                format_metrics,
                reward_format_metrics,
            ) = self.verify(data)
            reward_metrics.update(verifier_metrics)
        for i in range(verifier_reward.shape[0]):
            verifier_reward[i, valid_response_length[i] - 1] += verifier_score[i]

        reward_tensor_dict["gt_scores"] = verifier_reward
        reward_coef = 5.0
        if reward_coef != 0:
            reward_metrics["verifier"] = (
                reward_tensor_dict["gt_scores"].sum(dim=1).mean().item()
            )
            reward_tensor += reward_coef * reward_tensor_dict["gt_scores"]

        reward_tensor_dict["all"] = reward_tensor
        reward_metrics["reward_all"] = reward_tensor.sum(dim=-1).mean(dim=0).item()
        return reward_tensor_dict, reward_metrics


@dataclass
class RLTrainerConfig(TrainerConfig):
    clip_ratio_high: float = field(default=0.28)
    clip_ratio_low: float = field(default=0.2)
    entropy_coeff: float = field(default=0.0)
    value_loss_coeff: float = field(default=0.5)
    max_grad_norm: float = field(default=1.0)
    ppo_epochs: int = field(default=4)
    ppo_mini_batch_size: int = field(default=128)
    rollout_batch_size: int = field(default=32)
    max_rollout_length: int = field(default=128)
    eval_episodes: int = field(default=10)
    eval_frequency: int = field(default=5)


class DexboticRLTrainer(DexboticTrainer):
    def __init__(self, *args, **kwargs):
        self.rl_config: GRPOConfig = kwargs.pop("rl_config", GRPOConfig())
        self.env_config: RLEnvironmentConfig = kwargs.pop(
            "env_config", RLEnvironmentConfig()
        )
        super().__init__(*args, **kwargs)
        self.grpo_trainer = GRPOTrainer(self.rl_config, self.env_config)

    def generate_traj_mask(self, end_step, traj_len):
        steps = torch.arange(traj_len, device=end_step.device)
        steps_expanded = steps.unsqueeze(0).expand(end_step.size(0), -1)
        mask = steps_expanded < end_step.unsqueeze(1)
        return mask

    def apply_mask_with_grad_control(self, log_probs, entropy, mask):
        mask_expanded = mask.unsqueeze(-1)
        log_probs_masked = torch.where(
            mask_expanded, log_probs, torch.zeros_like(log_probs, requires_grad=False)
        )
        entropy_masked = torch.where(
            mask_expanded, entropy, torch.zeros_like(entropy, requires_grad=False)
        )
        return log_probs_masked, entropy_masked

    def _forward_micro_batch(
        self, micro_batch, temperature, masked
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = micro_batch["responses"].size(0)
        traj_len = micro_batch["responses"].size(1)
        tot_pad_len = micro_batch["input_ids"].size(2)

        assert all(
            micro_batch[key].size(0) == batch_size
            for key in ["responses", "input_ids", "attention_mask", "pixel_values"]
        )
        assert all(
            micro_batch[key].size(1) == traj_len
            for key in ["responses", "input_ids", "attention_mask", "pixel_values"]
        )
        assert all(
            micro_batch[key].size(2) == tot_pad_len
            for key in ["input_ids", "attention_mask"]
        )
        if self.exp_config.model_config.use_proprio:
            assert (
                micro_batch["proprio"].size(0) == batch_size
                and micro_batch["proprio"].size(1) == traj_len
                and micro_batch["proprio"].size(2) == self.config.action_token_len
            )

        response_length = micro_batch["responses"].size(-1)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            attention_mask = micro_batch["attention_mask"]
            pixel_values = micro_batch["pixel_values"]
            responses = micro_batch["responses"]

            input_ids = input_ids.reshape(
                (batch_size * traj_len,) + input_ids.shape[2:]
            )
            attention_mask = attention_mask.reshape(
                (batch_size * traj_len,) + attention_mask.shape[2:]
            )
            pixel_values = pixel_values.reshape(
                (batch_size * traj_len,) + pixel_values.shape[2:]
            )
            responses = responses.reshape(
                (batch_size * traj_len,) + responses.shape[2:]
            )

            if self.exp_config.model_config.use_proprio:
                proprio = micro_batch["proprio"]
                proprio = proprio.reshape((batch_size * traj_len,) + proprio.shape[2:])
            else:
                proprio = None

            logits = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                images=pixel_values,
                states=proprio,
            ).logits

            num_bins = self.exp_config.model_config.model.num_bins
            action_chunks_len = self.model.config.chunk_size
            action_token_len = self.model.config.action_dim

            logits = logits[..., -num_bins + 1 :]
            responses = responses - (len(self.processing_class) - num_bins + 1)
            logits = logits.div(temperature)

            log_probs = logprobs_from_logits(logits, responses)
            entropy = entropy_from_logits(logits)

            assert len(log_probs.shape) == 2 and len(entropy.shape) == 2
            log_probs = log_probs.reshape(
                (batch_size, traj_len * action_chunks_len, action_token_len)
            )
            entropy = entropy.reshape(
                (batch_size, traj_len * action_chunks_len, action_token_len)
            )
            if masked:
                mask = self.generate_traj_mask(
                    micro_batch["finish_step"], traj_len * action_chunks_len
                )
                log_probs, entropy = self.apply_mask_with_grad_control(
                    log_probs, entropy, mask
                )
                log_probs = log_probs.reshape((batch_size, traj_len * response_length))
                entropy = entropy.reshape((batch_size, traj_len * response_length))

        return entropy, log_probs

    def _forward_micro_batch_update(
        self, input_ids, attention_mask, pixel_values, responses, temperature, proprio
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                images=pixel_values,
                states=proprio,
            ).logits
            assert logits.requires_grad
            num_bins = self.exp_config.model_config.model.num_bins
            logits = logits[..., -num_bins + 1 :]
            responses = responses - (len(self.processing_class) - num_bins + 1)
            logits = logits.div(temperature)
            log_probs = logprobs_from_logits(logits, responses)
            entropy = entropy_from_logits(logits)
            log_probs = log_probs.reshape((1, -1))
            entropy = entropy.reshape((1, -1))
            return entropy, log_probs

    def compute_log_prob(self, data, masked=True) -> Tuple[torch.Tensor, torch.Tensor]:
        self.model.eval()
        temperature = self.exp_config.model_config.rollout.temperature
        self.pad_token_id = self.processing_class.pad_token_id
        select_keys = [
            "responses",
            "input_ids",
            "attention_mask",
            "pixel_values",
            "finish_step",
        ]
        if self.exp_config.model_config.use_proprio:
            select_keys.append("proprio")
        with torch.no_grad():
            entropy, log_probs = self._forward_micro_batch(
                data, temperature=temperature, masked=masked
            )
        return entropy, log_probs

    def compute_loss(self, model, inputs, return_outputs=False, *args, **kwargs):
        old_log_probs = inputs.get("old_log_probs")
        rewards = inputs.get("rewards")
        values = inputs.get("values")
        eos_mask = inputs.get("eos_mask")
        prompt_indices = inputs.get("prompt_indices")

        if old_log_probs is None or rewards is None or eos_mask is None:
            return super().compute_loss(model, inputs, return_outputs, *args, **kwargs)

        outputs = model(
            **{
                k: v
                for k, v in inputs.items()
                if k
                not in [
                    "old_log_probs",
                    "rewards",
                    "values",
                    "eos_mask",
                    "prompt_indices",
                ]
            }
        )

        logits = outputs.logits
        labels = inputs["labels"]
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        log_probs = torch.log_softmax(shift_logits, dim=-1)
        current_log_probs = torch.gather(
            log_probs, -1, shift_labels.unsqueeze(-1)
        ).squeeze(-1)

        advantages, returns = self.grpo_trainer.compute_advantages(
            rewards=rewards, values=values, eos_mask=eos_mask, index=prompt_indices
        )

        policy_loss, clip_frac, approx_kl = self.grpo_trainer.compute_policy_loss(
            old_log_prob=old_log_probs,
            log_prob=current_log_probs,
            advantages=advantages,
            eos_mask=eos_mask,
            clip_ratio_high=self.exp_config.trainer_config.clip_ratio_high,
            clip_ratio_low=self.exp_config.trainer_config.clip_ratio_low,
        )

        entropy_loss = self._compute_entropy_loss(shift_logits, eos_mask)
        total_loss = (
            policy_loss + self.exp_config.trainer_config.entropy_coeff * entropy_loss
        )

        self._log_rl_metrics(
            {
                "policy_loss": policy_loss.item(),
                "entropy_loss": entropy_loss.item(),
                "clip_frac": clip_frac.item(),
                "approx_kl": approx_kl.item(),
                "total_loss": total_loss.item(),
            }
        )

        if return_outputs:
            return total_loss, outputs
        else:
            return total_loss

    def _compute_policy_loss(
        self,
        old_log_prob,
        log_prob,
        advantages,
        eos_mask,
        clip_ratio_high,
        clip_ratio_low,
    ):
        negative_approx_kl = log_prob - old_log_prob
        ratio = torch.exp(negative_approx_kl)
        ppo_kl = masked_mean(-negative_approx_kl, eos_mask)
        pg_losses = -advantages * ratio
        pg_losses2 = -advantages * torch.clamp(
            ratio, 1.0 - clip_ratio_low, 1.0 + clip_ratio_high
        )
        pg_loss = masked_mean(torch.max(pg_losses, pg_losses2), eos_mask)
        pg_clipfrac = masked_mean(torch.gt(pg_losses2, pg_losses).float(), eos_mask)
        return pg_loss, pg_clipfrac, ppo_kl

    def _log_rl_metrics(self, metrics: Dict[str, float]) -> None:
        if self.state.global_step % self.args.logging_steps == 0:
            for key, value in metrics.items():
                logger.info(f"Step {self.state.global_step}: {key} = {value:.6f}")

    def batch_segment(
        self, batch: Dict[str, Any], original_batch_size, num_segment: int = 1
    ):
        if num_segment == 1:
            return [batch]
        try:
            original_batch_size = int(original_batch_size)
        except Exception:
            pass
        rollout_num = original_batch_size // num_segment
        batch_out = []
        random_index = random.sample(range(original_batch_size), original_batch_size)
        for n in range(num_segment):
            batch_temp = {}
            for keys in batch:
                if n != num_segment - 1:
                    batch_temp[keys] = batch[keys][
                        random_index[n * rollout_num : (n + 1) * rollout_num]
                    ]
                else:
                    batch_temp[keys] = batch[keys][random_index[n * rollout_num :]]
            batch_out.append(batch_temp)
        return batch_out

    def update_policy(self, batch: Dict[str, Any]) -> Dict[str, float]:
        self.model.train()
        if self.optimizer is None:
            self.create_optimizer()

        original_batch_size = batch["responses"].size(0)
        loss_info = {"actor/pg_loss": 0, "actor/pg_clipfrac": 0, "actor/ppo_kl": 0}

        for update_iter in range(3):
            batches = self.batch_segment(batch, original_batch_size, num_segment=1)
            seg_batch_size = batches[0]["responses"].size(0)

            for batch_seg in batches:
                self.optimizer.zero_grad()
                temperature = self.exp_config.model_config.rollout.temperature

                for batch_idx in range(seg_batch_size):
                    single_batch = self._extract_single_sample(batch_seg, batch_idx)
                    responses = single_batch["responses"]
                    response_length = responses.size(1) * responses.size(2)
                    finish_step = (
                        single_batch["finish_step"]
                        * self.exp_config.model_config.action_dim
                    )

                    steps = torch.arange(response_length, device=responses.device)
                    steps_expanded = steps.unsqueeze(0).expand(
                        single_batch["responses"].size(0), -1
                    )
                    response_mask = steps_expanded < finish_step.unsqueeze(1)
                    response_mask_sum = response_mask.sum()

                    if response_mask_sum == 0:
                        continue

                    old_log_prob = single_batch["old_log_probs"]
                    advantages = single_batch["advantages"]
                    clip_ratio_high = self.exp_config.model_config.actor.clip_ratio_high
                    clip_ratio_low = self.exp_config.model_config.actor.clip_ratio_low

                    batch_size, traj_len = single_batch["responses"].size(
                        0
                    ), single_batch["responses"].size(1)
                    reshaped_tensors = self._reshape_tensors_for_forward(
                        single_batch, batch_size, traj_len
                    )

                    traj_mini_batch_size = (
                        self.exp_config.model_config.actor.traj_mini_batch_size
                    )
                    traj_split_num = max(1, int(traj_len / traj_mini_batch_size))

                    for i in range(0, traj_len, int(traj_len / traj_split_num)):
                        chunk_end = min(i + int(traj_len / traj_split_num), traj_len)
                        entropy, log_prob = self._forward_micro_batch_update(
                            input_ids=reshaped_tensors["input_ids"][i:chunk_end],
                            attention_mask=reshaped_tensors["attention_mask"][
                                i:chunk_end
                            ],
                            pixel_values=reshaped_tensors["pixel_values"][i:chunk_end],
                            responses=reshaped_tensors["responses"][i:chunk_end],
                            temperature=temperature,
                            proprio=reshaped_tensors["proprio"][i:chunk_end]
                            if reshaped_tensors["proprio"] is not None
                            else None,
                        )

                        slice_start = (
                            i
                            * self.exp_config.model_config.action_dim
                            * self.exp_config.model_config.chunk_size
                        )
                        slice_end = (
                            chunk_end
                            * self.exp_config.model_config.action_dim
                            * self.exp_config.model_config.chunk_size
                        )

                        old_log_prob_chunk = old_log_prob[:, slice_start:slice_end]
                        advantages_chunk = advantages[:, slice_start:slice_end]
                        response_mask_chunk = response_mask[:, slice_start:slice_end]

                        pg_loss, pg_clipfrac, ppo_kl = self._compute_policy_loss(
                            old_log_prob=old_log_prob_chunk,
                            log_prob=log_prob,
                            advantages=advantages_chunk,
                            eos_mask=response_mask_chunk,
                            clip_ratio_high=clip_ratio_high,
                            clip_ratio_low=clip_ratio_low,
                        )

                        chunk_mask_sum = response_mask_chunk.sum()
                        if chunk_mask_sum > 0:
                            pg_loss_normalized = (
                                pg_loss * chunk_mask_sum / response_mask_sum
                            )
                            policy_loss = pg_loss_normalized / original_batch_size
                            policy_loss.backward()

                            loss_info["actor/pg_loss"] += (
                                pg_loss_normalized.detach().item() / original_batch_size
                            )
                            loss_info["actor/pg_clipfrac"] += (
                                pg_clipfrac * chunk_mask_sum / response_mask_sum
                            ).detach().item() / original_batch_size
                            loss_info["actor/ppo_kl"] += (
                                ppo_kl * chunk_mask_sum / response_mask_sum
                            ).detach().item() / original_batch_size

                        del entropy, log_prob, pg_loss, pg_clipfrac, ppo_kl
                        del old_log_prob_chunk, advantages_chunk, response_mask_chunk
                        if chunk_mask_sum > 0:
                            del policy_loss

                    del (
                        reshaped_tensors,
                        old_log_prob,
                        advantages,
                        response_mask,
                        steps,
                        single_batch,
                    )
                    torch.cuda.empty_cache()

                self._apply_gradients()

        loss_info["actor/grad_norm"] = self._get_grad_norm()
        torch.cuda.empty_cache()
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return loss_info

    def _extract_single_sample(
        self, batch_seg: Dict[str, Any], batch_idx: int
    ) -> Dict[str, Any]:
        single_batch = {
            "responses": batch_seg["responses"][batch_idx : batch_idx + 1],
            "input_ids": batch_seg["input_ids"][batch_idx : batch_idx + 1],
            "attention_mask": batch_seg["attention_mask"][batch_idx : batch_idx + 1],
            "pixel_values": batch_seg["pixel_values"][batch_idx : batch_idx + 1],
            "finish_step": batch_seg["finish_step"][batch_idx : batch_idx + 1],
            "old_log_probs": batch_seg["old_log_probs"][batch_idx : batch_idx + 1],
            "advantages": batch_seg["advantages"][batch_idx : batch_idx + 1],
        }
        if self.exp_config.model_config.use_proprio:
            single_batch["proprio"] = batch_seg["proprio"][batch_idx : batch_idx + 1]
        return single_batch

    def _reshape_tensors_for_forward(
        self, single_batch: Dict[str, Any], batch_size: int, traj_len: int
    ) -> Dict[str, Any]:
        reshaped = {
            "input_ids": single_batch["input_ids"].reshape(
                (batch_size * traj_len,) + single_batch["input_ids"].shape[2:]
            ),
            "attention_mask": single_batch["attention_mask"].reshape(
                (batch_size * traj_len,) + single_batch["attention_mask"].shape[2:]
            ),
            "pixel_values": single_batch["pixel_values"].reshape(
                (batch_size * traj_len,) + single_batch["pixel_values"].shape[2:]
            ),
            "responses": single_batch["responses"].reshape(
                (batch_size * traj_len,) + single_batch["responses"].shape[2:]
            ),
            "proprio": None,
        }
        if self.exp_config.model_config.use_proprio:
            reshaped["proprio"] = single_batch["proprio"].reshape(
                (batch_size * traj_len,) + single_batch["proprio"].shape[2:]
            )
        return reshaped

    def _apply_gradients(self):
        if self.args.max_grad_norm is not None and self.args.max_grad_norm > 0:
            if hasattr(self, "accelerator") and self.accelerator is not None:
                grad_norm = self.accelerator.clip_grad_norm_(
                    self.model.parameters(), self.args.max_grad_norm
                )
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.args.max_grad_norm
                )
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), float("inf")
            )
        self.optimizer.step()
        self._grad_norm = grad_norm
        return grad_norm

    def _get_grad_norm(self):
        return (
            self._grad_norm.detach().item()
            if isinstance(self._grad_norm, torch.Tensor)
            else self._grad_norm
        )

    def _save_checkpoint(self, model, trial, metrics=None) -> None:
        from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

        logger.info(f"Saving RL checkpoint at step {self.state.global_step}")
        self.is_deepspeed_enabled = False
        super()._save_checkpoint(model, trial, metrics)

        if self.args.local_rank == 0 or self.args.local_rank == -1:
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            rl_training_state = {
                "global_step": self.state.global_step,
                "rl_config": self.rl_config,
                "env_config": self.env_config,
            }
            rl_state_path = os.path.join(output_dir, "rl_training_state.pt")
            torch.save(rl_training_state, rl_state_path)
            logger.info(f"Saved RL training state to {rl_state_path}")

        torch.cuda.synchronize()
        torch.distributed.barrier()
        torch.cuda.empty_cache()
        return metrics
