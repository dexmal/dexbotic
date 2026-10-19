from typing import Union, Tuple, Dict, List, Optional
import math
from contextlib import contextmanager
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import ModelOutput

from dexbotic.model.memvla.action_model.builder import build_action_model
from dexbotic.model.dexbotic_arch import (ActionOutputForCausalLM,
                                          CausalLMOutputDexbotic,
                                          DexboticConfig, DexboticForCausalLM,
                                          DexboticVLMModel)

KeyT = Union[Tuple[int, int], Tuple[int, int, int]]


class MemVLAConfig(DexboticConfig):
    model_type = "dexbotic_memvla"
    action_model_type: Optional[str] = None
    action_dim: Optional[int] = None
    chunk_size: Optional[int] = None


@dataclass
class CausalLMOutputDexbotic(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    vision_proj_feats: Optional[torch.Tensor] = None


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0,
                                                 end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(
            t, self.frequency_embedding_size).to(
            self.mlp[0].weight.dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


class CrossTransformerBlock(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int = 4, dropout: float = 0.1,):
        super().__init__()
        assert feature_dim % num_heads == 0, "feature_dim % num_heads must be 0"
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        self.feature_dim = feature_dim
        self.dropout = dropout

        # QKV projection
        self.q_proj = nn.Linear(feature_dim, feature_dim)
        self.k_proj = nn.Linear(feature_dim, feature_dim)
        self.v_proj = nn.Linear(feature_dim, feature_dim)

        self.attn_norm = nn.LayerNorm(feature_dim)

        # Feed-Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 4, feature_dim),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(feature_dim)

    def forward(self,
                query: torch.Tensor,  # (B, N, D)
                k: torch.Tensor,      # (B, M, D)
                v: torch.Tensor       # (B, M, D)
                ) -> torch.Tensor:
        B, N, D = query.shape
        _, M, _ = k.shape

        # Q,K,V projection
        q = self.q_proj(query).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, h, N, d_h)
        k = self.k_proj(k).reshape(B, M, self.num_heads, self.head_dim).transpose(1, 2)      # (B, h, M, d_h)
        v = self.v_proj(v).reshape(B, M, self.num_heads, self.head_dim).transpose(1, 2)      # (B, h, M, d_h)

        # Multi-head attention
        attn_out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout, is_causal=False
        )  # (B, h, N, d_h)

        attn_out = attn_out.transpose(1, 2).reshape(B, N, D)  # 合并头

        # 残差 + LN
        x = self.attn_norm(query + attn_out)

        # FFN + 残差 + LN
        ffn_out = self.ffn(x)
        return self.ffn_norm(x + ffn_out)


class BottleneckSE(nn.Module):
    def __init__(self, C_in, C_out, reduction=16, hidden_ratio=0.5):
        super().__init__()
        self.C_in = C_in
        self.C_out = C_out

        hidden_se = max(1, C_in // reduction)

        self.excite = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(C_in, hidden_se, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_se, C_in, 1, bias=True),
            nn.Sigmoid(),
        )

        hidden_mlp = max(1, int(C_in * hidden_ratio))

        self.reduce = nn.Sequential(
            nn.Conv2d(C_in, hidden_mlp, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_mlp, C_out, 1, bias=True),
        )

    def forward(self, x):
        _b, _n, _c = x.shape
        _h = _w = int(math.sqrt(_n))
        assert _h * _h == _n, "Input feature has no spatial structure"

        x = x.reshape(_b, _h, _w, _c).permute(0, 3, 1, 2).contiguous()  # (B, C_in, H, W)

        w = self.excite(x)  # (B, C_in, 1, 1)
        x = x * w

        out = self.reduce(x)
        out = out.reshape(_b, self.C_out, _n).permute(0, 2, 1).contiguous()

        return out


class GateFusion(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(dim * 2, dim)
        nn.init.normal_(self.proj.weight, mean=0.0, std=1e-3)
        nn.init.normal_(self.proj.bias, mean=0.0, std=1e-3)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        scale = torch.sigmoid(
            self.proj(
                torch.cat([x1, x2],
                dim=-1)
            )
        )

        fused = scale * x1 + (1 - scale) * x2
        return fused


class PerCogMemBank(nn.Module):
    def __init__(
        self,
        dataloader_type: str,
        group_size: int,
        per_token_size: int,
        cog_token_size: int,
        mem_length: int = 16,
        retrieval_layers: int = 2,
        use_timestep_pe: bool = True,
        fusion_type: str = 'gate',
        consolidate_type: str = 'tome',
        update_fused: bool = True,
    ):
        super().__init__()
        assert dataloader_type in ('stream', 'group', 'parallel_stream')
        assert fusion_type in ('gate', 'add')
        assert consolidate_type in ('fifo', 'tome')

        self.roles = ('per', 'cog')
        self.dataloader_type = dataloader_type
        self.group_size = group_size
        self.mem_length = mem_length
        self.retrieval_layers = retrieval_layers
        self.use_timestep_pe = use_timestep_pe
        self.fusion_type = fusion_type
        self.consolidate_type = consolidate_type
        self.update_fused = update_fused

        self.token_dim: Dict[str, int] = {
            'per': per_token_size,
            'cog': cog_token_size,
        }

        self.retrieval_blocks = nn.ModuleDict({
            r: nn.ModuleList([CrossTransformerBlock(self.token_dim[r])
                              for _ in range(self.retrieval_layers)])
            for r in self.roles
        })

        if self.fusion_type == 'gate':
            self.gate_fusion_blocks = nn.ModuleDict({
                r: GateFusion(self.token_dim[r]) for r in self.roles
            })

        if self.use_timestep_pe:
            self.timestep_embedders = nn.ModuleDict({
                r: TimestepEmbedder(
                    self.token_dim[r],
                ) for r in self.roles
            })
        else:
            self.timestep_embedders = None

        self.reset()

    def reset(self):
        # banks[role][episode_id] = [(timestep, feat[N,D]), ...]
        self.banks: Dict[str, Dict[KeyT, List[Tuple[torch.Tensor, torch.Tensor]]]] = {
            r: {} for r in self.roles
        }

        self.prev_eids: Dict[str, Dict[int, KeyT]] = {r: {} for r in self.roles}
        self.eid_stream: Dict[str, Optional[Tuple[int, int]]] = {r: None for r in self.roles}

    def clear_episode(self, role: str, episode_id: KeyT):
        self.banks[role].pop(episode_id, None)

    @torch.no_grad()
    def _consolidate_with_token_merge(self, role: str, episode_id: KeyT):
        bank = self.banks[role].get(episode_id, [])
        T = len(bank)
        if T < 2:
            return

        feats = [feat for (_, feat) in bank]

        sims = []
        for i in range(T - 1):
            f1 = feats[i].flatten(1) if feats[i].dim() > 1 else feats[i].unsqueeze(0)
            f2 = feats[i+1].flatten(1) if feats[i+1].dim() > 1 else feats[i+1].unsqueeze(0)
            sims.append(F.cosine_similarity(f1, f2, dim=1).mean().item())

        idx_max = int(torch.tensor(sims).argmax().item())

        timestep_i, feat_i = bank[idx_max]
        timestep_j, feat_j = bank[idx_max + 1]
        fused_feat = 0.5 * (feat_i + feat_j)
        fused_timestep = 0.5 * (timestep_i + timestep_j) if timestep_i is not None else None

        bank[idx_max] = (fused_timestep, fused_feat.detach().clone())
        bank.pop(idx_max + 1)

    @torch.no_grad()
    def _memory_consolidate(
            self,
            role: str,
            episode_id: KeyT,
            feat: torch.Tensor,
            timestep: Optional[torch.Tensor]):
        if episode_id not in self.banks[role]:
            self.banks[role][episode_id] = []

        self.banks[role][episode_id].append((timestep, feat.detach().clone()))

        while len(self.banks[role][episode_id]) > self.mem_length:
            if self.consolidate_type == 'fifo':
                self.banks[role][episode_id] = self.banks[role][episode_id][-self.mem_length:]
            elif self.consolidate_type == "tome":
                self._consolidate_with_token_merge(role, episode_id)
            else:
                raise NotImplementedError

    def _encode_time(self, role: str, timestep: torch.Tensor) -> torch.Tensor:
        assert self.use_timestep_pe

        return self.timestep_embedders[role](timestep)

    def _process_batch(
        self,
        role: str,
        tokens: torch.Tensor, # [B, N, D_role]
        episode_ids: List[Tuple[int, int]],
        timesteps: List[torch.Tensor],
    ) -> torch.Tensor:
        assert role in self.roles
        assert episode_ids is not None, "episode_ids must be provided during training"

        if self.use_timestep_pe:
            assert timesteps is not None, "timesteps must be provided during training"

        B, N, D = tokens.shape
        outputs = []

        if self.training:
            if self.dataloader_type == 'group':
                self.banks[role].clear()
                self.prev_eids[role].clear()
                self.eid_stream[role] = None
            elif self.dataloader_type == 'stream':
                first_eid = episode_ids[0]
                prev_active = self.eid_stream[role]
                if prev_active is not None and prev_active != first_eid:
                    self.clear_episode(role, prev_active)
                self.eid_stream[role] = first_eid
            elif self.dataloader_type == 'parallel_stream':
                episode_ids = [(i, eid[0], eid[1]) for i, eid in enumerate(episode_ids)]
        else:
            if self.dataloader_type == 'group' or self.dataloader_type == 'stream':
                episode_ids = [(0, 0) for _ in range(B)]
            elif self.dataloader_type == 'parallel_stream':
                episode_ids = [(i, 0, 0) for i in range(B)]

        for i in range(B):
            # 1) episode management
            eid = episode_ids[i]
            if self.training:
                if self.dataloader_type == 'stream':
                    if i > 0 and episode_ids[i] != episode_ids[i - 1]:
                        self.clear_episode(role, episode_ids[i - 1])
                        self.eid_stream[role] = episode_ids[i]
                elif self.dataloader_type == 'parallel_stream':
                    prev = self.prev_eids[role].get(i, None)
                    if prev is not None and prev != eid:
                        self.clear_episode(role, prev)

                    self.prev_eids[role][i] = eid

            # 2) memory retrieval
            working_mem = tokens[i].unsqueeze(0)  # (1, N, D)

            hist = self.banks[role].get(eid, [])
            if len(hist) > 0:
                hist_feats = [feat for _, feat in hist]
                episode_mem = torch.stack(hist_feats, dim=0).reshape(-1, D).unsqueeze(0)  # (1, T*N, D)

                if self.use_timestep_pe:
                    hist_timesteps = [t for t, _ in hist]
                    hist_timesteps = torch.stack(hist_timesteps, dim=0).to(working_mem.device)
                    pe = self._encode_time(role, hist_timesteps).unsqueeze(0)  # (1, T, D)
                    pe = pe.repeat_interleave(N, dim=1) # (1, T*N, D)
                else:
                    pe = torch.zeros_like(episode_mem)
            else:
                # without history：working memory as episode memory
                episode_mem = working_mem  # (1, N, D)

                if self.use_timestep_pe:
                    t_now = timesteps[i].reshape(1).to(working_mem.device)
                    pe = self._encode_time(role, t_now).unsqueeze(0)  # (1, 1, D)
                    pe = pe.repeat_interleave(N, dim=1)  # (1, N, D)
                else:
                    pe = torch.zeros_like(episode_mem)

            query = working_mem
            for block in self.retrieval_blocks[role]:
                query = block(query, episode_mem + pe, episode_mem)

            retrieved_episode_mem = query

            # 3) memory adaptive fusion
            if self.fusion_type == 'add':
                fused_feats = (working_mem + retrieved_episode_mem) * 0.5
            elif self.fusion_type == 'gate':
                fused_feats = self.gate_fusion_blocks[role](working_mem, retrieved_episode_mem)

            outputs.append(fused_feats)

            # 4) memory consolidate
            timestep_i = timesteps[i] if self.use_timestep_pe else None

            if self.update_fused:
                self._memory_consolidate(role, eid, fused_feats.squeeze(0), timestep_i)
            else:
                self._memory_consolidate(role, eid, tokens[i], timestep_i)

        return torch.cat(outputs, dim=0)  # [B, N, D_role]

    def process_batch_per(
        self,
        per_tokens: torch.Tensor, # [B, N, D_per]
        episode_ids: List[Tuple[int, int]],
        timesteps: List[torch.Tensor],
    ) -> torch.Tensor:
        return self._process_batch('per', per_tokens, episode_ids, timesteps)

    def process_batch_cog(
        self,
        cog_tokens: torch.Tensor, # [B, N, D_cog]
        episode_ids: list[Tuple[int, int]],
        timesteps: List[torch.Tensor],
    ) -> torch.Tensor:
        return self._process_batch('cog', cog_tokens, episode_ids, timesteps)


class MemVLAModel(DexboticVLMModel):
    def __init__(self, config: MemVLAConfig):
        # init vision encoder + llm + projector
        super().__init__(config)

        # init perceptual compression module
        if getattr(config, "per_token_size", None) is not None:
            self.per_compr = self._build_per_compr_module(config)

        # init perceptual-cognitive memory module
        _need = ["dataloader_type","group_size","mem_length","retrieval_layers",
                 "use_timestep_pe","fusion_type","consolidate_type","per_token_size"]
        if all(getattr(config, k, None) is not None for k in _need):
            self.per_cog_mem_bank = self._build_per_cog_mem_bank_module(config)

        # init action model
        if config.action_model_type is not None:
            self.action_head = self._build_action_head_module(config)

    @property
    def action_head_module(self) -> nn.Module:
        return self.action_head

    @property
    def action_head_prefix(self) -> str:
        return 'action_head'

    def initialize_model(self, extra_config: dict):
        # rebuild modules if modules not exist in pretrained model
        for key, value in extra_config.items():
            setattr(self.config, key, value)

        self.mm_vision_tower = self._build_mm_vision_module(self.config.mm_vision_tower)
        self.mm_projector = self._build_mm_projector_module(self.config)
        self.per_compr = self._build_per_compr_module(self.config)
        self.per_cog_mem_bank = self._build_per_cog_mem_bank_module(self.config)
        self.action_head = self._build_action_head_module(self.config)

    def _build_per_compr_module(self, config: MemVLAConfig):
        if getattr(self, 'per_compr', None) is not None:
            return self.per_compr

        assert config.per_token_size is not None

        per_compr = BottleneckSE(
            C_in=config.hidden_size,
            C_out=config.per_token_size,
        )

        return per_compr

    def _build_per_cog_mem_bank_module(self, config: MemVLAConfig):
        if getattr(self, 'per_cog_mem_bank', None) is not None:
            return self.per_cog_mem_bank

        assert config.dataloader_type is not None
        assert config.group_size is not None
        assert config.per_token_size is not None
        assert config.mem_length is not None
        assert config.retrieval_layers is not None
        assert config.use_timestep_pe is not None
        assert config.fusion_type is not None
        assert config.consolidate_type is not None
        if not hasattr(config, "update_fused"):
            config.update_fused = True

        per_cog_mem_bank = PerCogMemBank(
            dataloader_type=config.dataloader_type,
            group_size=config.group_size,
            per_token_size=config.per_token_size,
            cog_token_size=config.hidden_size,
            mem_length=config.mem_length,
            retrieval_layers=config.retrieval_layers,
            use_timestep_pe=config.use_timestep_pe,
            fusion_type=config.fusion_type,
            consolidate_type=config.consolidate_type,
            update_fused=config.update_fused,
        )

        return per_cog_mem_bank

    def _build_action_head_module(self, config: MemVLAConfig):
        if getattr(self, 'action_head', None) is not None:
            if self.action_head.model_type != config.action_model_type:
                print(f"Warning: Rebuilding action model from {self.action_head.model_type} to {config.action_model_type}")
                self.action_head = build_action_model(config)
            else:
                assert config.per_token_size is not None
                if getattr(self.action_head.net, 'per_token_embedder', None) is not None:
                    assert self.action_head.net.blocks[0].per_attn is not None
                else:
                    new_action_head = build_action_model(config)

                    old_state_dict = self.action_head.state_dict()
                    new_state_dict = new_action_head.state_dict()
                    for k, v in old_state_dict.items():
                        assert k in new_state_dict
                        new_state_dict[k] = v

                    new_action_head.load_state_dict(new_state_dict, strict=True)
                    self.action_head = new_action_head
        else:
            self.action_head = build_action_model(config)
        return self.action_head


class MemVLAForCausalLM(DexboticForCausalLM, ActionOutputForCausalLM):
    config_class = MemVLAConfig

    def _real_init(self, config: MemVLAConfig):
        self.model = MemVLAModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

        self.cur_timestep = 0 # for inference

    def forward(self,
                input_ids: torch.LongTensor = None,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                past_key_values: Optional[List[torch.FloatTensor]] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                labels: Optional[torch.LongTensor] = None,
                use_cache: Optional[bool] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                images: Optional[torch.FloatTensor] = None,
                return_dict: Optional[bool] = None,
                cache_position: Optional[torch.LongTensor] = None,
                actions: Optional[torch.LongTensor] = None,
                states: Optional[torch.LongTensor] = None,
                repeated_diffusion_steps: int = 4,
                indexes: List[int] = None,
                **kwargs,
                ) -> CausalLMOutputDexbotic:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        with self.capture_projected_vision(self.model) as buf:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                cache_position
            ) = self.model._prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                cache_position,
                images
            )

        vision_proj_feats = buf.get("vision_proj")

        outputs = self.model.llm(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_hidden_states=True,
        )

        last_hidden_state = outputs.hidden_states[-1]

        if attention_mask is not None and actions is not None:
            # extract the cognition feature
            cumulative_sum = attention_mask.cumsum(dim=1)
            last_unmask_indices = (
                cumulative_sum == cumulative_sum.max(
                    dim=1, keepdim=True)[0]).float().argmax(
                dim=1)
            expanded_indices = last_unmask_indices.unsqueeze(
                -1).expand(-1, last_hidden_state.size(-1))
            cog_tokens = last_hidden_state.gather(
                1, expanded_indices.unsqueeze(1))  # [B, 1, D]

            per_tokens = self.model.per_compr(vision_proj_feats)

            episode_ids: List[Tuple[int, int]] = [tuple(item[:2]) for item in indexes]
            timesteps: List[torch.Tensor] = [torch.tensor(item[2], device=cog_tokens.device) for item in indexes]

            cog_tokens = self.model.per_cog_mem_bank.process_batch_cog(
                cog_tokens=cog_tokens,
                episode_ids=episode_ids,
                timesteps=timesteps,
            )

            per_tokens = self.model.per_cog_mem_bank.process_batch_per(
                per_tokens=per_tokens,
                episode_ids=episode_ids,
                timesteps=timesteps,
            )

        loss = None

        if actions is not None:
            actions = actions.reshape(actions.size(0), -
                                      1, self.config.action_dim).to(cog_tokens.dtype)
            actions_future = actions[:, :self.config.chunk_size, :]

            actions_repeated = actions_future.repeat(repeated_diffusion_steps, 1, 1)
            cog_tokens_repeated = cog_tokens.repeat(
                repeated_diffusion_steps, 1, 1)
            per_tokens_repeated = per_tokens.repeat(
                repeated_diffusion_steps, 1, 1)

            with torch.amp.autocast('cuda', dtype=torch.float32):
                loss = self.model.action_head_module.loss(
                    actions_repeated,
                    cog_tokens_repeated,
                    per_token=per_tokens_repeated,
                )

        if not return_dict:
            return (loss,) + last_hidden_state if loss is not None else last_hidden_state

        return CausalLMOutputDexbotic(
            loss=loss,
            logits=last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            vision_proj_feats=vision_proj_feats,
        )

    @torch.no_grad()
    def inference_action(self,
                         input_ids,
                         image_tensor,
                         episode_first_frame,
                         inference_args={},
                         **kwargs):
        cfg_scale = inference_args.get('cfg_scale', 1.5)
        num_ddim_steps = inference_args.get('num_ddim_steps', 10)
        action_norms = inference_args.get('action_norms')

        assert episode_first_frame in ('True', 'False'), "episode_first_frame must be 'True' or 'False'"
        if episode_first_frame == 'True':
            print(" ** reset memory ** ")
            self.model.per_cog_mem_bank.reset()
            self.cur_timestep = 0

        out_features = self.__call__(
            input_ids=input_ids,
            images=image_tensor,
            use_cache=True)

        cog_tokens = out_features.logits[:, -1, :].unsqueeze(1)  # [B, 1, D]

        per_tokens = self.model.per_compr(out_features.vision_proj_feats)

        episode_ids: List[Tuple[int, int]] = [(0, 0)]
        timesteps: List[torch.Tensor] = [torch.tensor(self.cur_timestep, device=cog_tokens.device)]
        self.cur_timestep += 1

        cog_tokens = self.model.per_cog_mem_bank.process_batch_cog(
            cog_tokens=cog_tokens,
            episode_ids=episode_ids,
            timesteps=timesteps,
        )
        per_tokens = self.model.per_cog_mem_bank.process_batch_per(
            per_tokens=per_tokens,
            episode_ids=episode_ids,
            timesteps=timesteps,
        )

        B = cog_tokens.size(0)

        noise = torch.randn(
            B,
            self.config.chunk_size,
            self.config.action_dim,
            device=cog_tokens.device,
            dtype=cog_tokens.dtype)  # [B T D]

        if cfg_scale > 1.0:
            noise = torch.cat([noise, noise], 0)

            uncondition = self.model.action_head.net.z_embedder.uncondition  # [1, D]
            uncondition = uncondition.unsqueeze(0).expand(B, 1, -1)  # [B, 1, D]
            z = torch.cat([cog_tokens, uncondition], 0)
            model_kwargs = dict(z=z, cfg_scale=cfg_scale)
            sample_fn = self.model.action_head.net.forward_with_cfg
        else:
            model_kwargs = dict(z=cog_tokens)
            sample_fn = self.model.action_head.net.forward

        model_kwargs.update({'per_token': per_tokens.repeat(2, 1, 1)})

        if self.model.action_head.ddim_diffusion is None:
            self.model.action_head.create_ddim(ddim_step=num_ddim_steps)

        samples = self.model.action_head.ddim_diffusion.ddim_sample_loop(
            sample_fn,
            noise.shape,
            noise,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            progress=False,
            device=cog_tokens.device,
            eta=0.0)
        if cfg_scale > 1.0:
            samples, _ = samples.chunk(2, dim=0)

        actions = self._denorm(samples[0].cpu().numpy(), action_norms).tolist()
        return actions

    @contextmanager
    def capture_projected_vision(self, model):
        buf = {}

        def _hook(_, __, output):
            buf["vision_proj"] = output

        h = model.mm_projector.register_forward_hook(_hook)
        try:
            yield buf
        finally:
            h.remove()
