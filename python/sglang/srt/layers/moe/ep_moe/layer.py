import logging
from typing import Callable, List, Optional, Tuple

import torch
from torch.nn import Module

from sglang.srt.layers.quantization.deep_gemm import _ENABLE_JIT_DEEPGEMM
from sglang.srt.managers.expert_location import get_global_expert_location_metadata
from sglang.srt.managers.expert_location_dispatch import ExpertLocationDispatchInfo
from sglang.srt.managers.schedule_batch import global_server_args_dict

try:
    from deep_gemm import (
        get_col_major_tma_aligned_tensor,
        m_grouped_gemm_fp8_fp8_bf16_nt_contiguous,
        m_grouped_gemm_fp8_fp8_bf16_nt_masked,
    )
    from sgl_kernel import silu_and_mul

    from sglang.srt.layers.quantization.fp8_kernel import (
        sglang_per_token_group_quant_fp8,
    )

    use_deep_gemm = True
except ImportError:
    use_deep_gemm = False

from sglang.srt.custom_op import CustomOp
from sglang.srt.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from sglang.srt.layers.moe.ep_moe.kernels import (
    ep_gather,
    ep_scatter,
    gelu_and_mul_triton_kernel,
    grouped_gemm_triton,
    post_reorder_triton_kernel,
    pre_reorder_triton_kernel,
    run_moe_ep_preproess,
    silu_and_mul_masked_post_quant_fwd,
    silu_and_mul_triton_kernel,
    tma_align_input_scale,
)
from sglang.srt.layers.moe.fused_moe_triton import FusedMoeWeightScaleSupported
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE, FusedMoEMethodBase
from sglang.srt.layers.moe.topk import select_experts
from sglang.srt.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8MoEMethod
from sglang.srt.layers.quantization.fp8_kernel import (
    scaled_fp8_quant,
    sglang_per_token_quant_fp8,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.utils import DeepEPMode, dispose_tensor, is_hip, set_weight_attrs

_is_hip = is_hip()

if _is_hip:
    from vllm._custom_ops import scaled_fp8_quant

logger = logging.getLogger(__name__)


class GroupedGemmRunner(torch.nn.Module):
    flashinfer_gemm_warpper = None

    def __init__(
        self,
        device,
        use_flashinfer: bool = False,
        use_per_token_if_dynamic: bool = True,
    ):
        super().__init__()
        self.device = device
        self.use_flashinfer = use_flashinfer
        self.use_per_token_if_dynamic = use_per_token_if_dynamic
        if self.use_flashinfer and GroupedGemmRunner.flashinfer_gemm_warpper is None:
            GroupedGemmRunner._init_flashinfer_wrapper(device)

    @classmethod
    def _init_flashinfer_wrapper(cls, device):
        from flashinfer import SegmentGEMMWrapper

        workspace_buffer = torch.empty(
            128 * 1024 * 1024, dtype=torch.int8, device=device
        )
        cls.flashinfer_gemm_warpper = SegmentGEMMWrapper(workspace_buffer)

    # c = a * b
    def forward(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        c: torch.Tensor,
        batch_size: int,
        weight_column_major: bool,
        seg_indptr: Optional[torch.Tensor] = None,
        weight_indices: Optional[torch.Tensor] = None,
        use_fp8_w8a8: bool = False,
        scale_a: torch.Tensor = None,
        scale_b: torch.Tensor = None,
        block_shape: Optional[List[int]] = None,
        c_dtype=None,
    ):
        if self.use_flashinfer:
            # TODO: flashinfer
            assert False
            assert GroupedGemmRunner.flashinfer_gemm_warpper is not None
            c = GroupedGemmRunner.flashinfer_gemm_warpper.run(
                x=a,
                weights=b,
                batch_size=batch_size,
                weight_column_major=weight_column_major,
                seg_indptr=seg_indptr,
                weight_indices=weight_indices,
            )
        else:
            # <NT> 分组矩阵乘，暂时只支持权重为列主序的情况，即可以内存连续地逐列处理B矩阵。
            # 因为nn.Linear的权重格式是(out_features, in_features), 
            # 正常GEMM中应该是A(batchs, in_features) * B(in_features, out_features) = Y(batchs, out_features)
            # B矩阵与nn.Linear权重矩阵呈转置关系，即行优先的B矩阵等同于列优先的nn.Linear权重矩阵。
            # seg_indptr是分段指针
            assert weight_column_major == True
            c = grouped_gemm_triton(
                a,
                b,
                c,
                batch_size,
                weight_column_major,
                seg_indptr,
                weight_indices,
                use_fp8_w8a8,
                scale_a,
                scale_b,
                block_shape=block_shape,
                c_dtype=c_dtype,
                use_per_token_if_dynamic=self.use_per_token_if_dynamic,
            )
        return c

# <NT> ep的pr: https://github.com/sgl-project/sglang/pull/2203
#      mla dp的pr: https://github.com/sgl-project/sglang/pull/1970
#      博客: https://mp.weixin.qq.com/s/hRVpCFynybW37jogW9_BXA
# 专家并行的MoE实现方式，与fused_moe_triton里的FusedMoE相对应。（enable_ep_moe则用EPMoE, 否则使用FusedMoE）
# 专家并行时，专家网络会分布在不同设备中。与fused_moe_triton中的FusedMoE是平级互替关系。
#
# num_experts: 专家总数
# top_k: 每个token选择的专家数量
# hidden_size: 隐藏层大小
# intermediate_size: 中间层大小，通常大于hidden_size
# params_dtype: 参数数据类型,默认为None使用系统默认类型
# renormalize: 是否重新归一化,默认True
# use_grouped_topk: 是否使用分组topk,默认False
# num_expert_group: 专家组数量,仅在use_grouped_topk=True时使用
# topk_group: 每组选择的专家数量,仅在use_grouped_topk=True时使用
# quant_config: 量化配置，支持Fp8EPMoEMethod
# tp_size: 张量并行大小（这里实质是专家并行大小，即总共有多少个设备）
# prefix: 前缀 (这里没使用到)
# correction_bias: 修正偏置, 用于对门控网路的输出进行调整后再选专家，用于biased_grouped_topk，与grouped_topk相对应
# 
# 通信问题: 
#   原来的ep并行训练中，需要使用两次all2all，第一次是将数据路由到合适的专家，第二次是将专家结果收集并重新分配。
# 第一次将数据路由到合适的专家，专家只会收到对应的数据，使用all2all精准定位分配。使用all gather通信量会更大。
# 第二次收集专家结果，因为只有部分专家有结果，所以也使用all2all，而不用all gather，通信量会少一些。
# 另外如某个token会在多个节点的某些专家上计算，最终是需要叠加专家结果。这里的叠加放到了all2all的数据汇总之后做。
#   这里的使用是使用allreduce代替两次all2all，为什么？
#   个人理解：这里每个节点都有所有输入数据，都会执行一遍门控网络去选择专家。都会有一份完整的专家选取结果和完整的输入数据。
# 每个节点只有部分专家，会对自己管理的专家所需要负责的数据从这份完整的输入数据中提取和整理并计算。
# 这样每个节点有n个专家关于m份数据的输出结果，然后多个节点的不同专家所负责的数据会有重叠，如一小份数据a，需要同时进入1，2，3节点的部分专家里进行计算。
# 1，2，3节点里部分专家对于数据a的计算结果需要加权叠加，并且没有参与计算的专家输出部分会被设置成0，可以直接使用all reduce。all reduce后不需要再做叠加操作。
#   
#   两个all2all换成1个allreduce，
# 第一个all2all的关键点在于原版ep训练时，门控网络不是每个节点都运行，需要all2all分发数据，而这里的ep并行，门控网络是每个节点都计算的，每个节点都拥有完整的数据。
# 第二个all2all是一小份数据在多个节点上的某些专家的计算结果是需要整合的，但是整合方式可能会随训练变化多样（如加权叠加，最大值，均值等），所以训练方案里是做all2all之后，再进行特定需要的叠加。
#             而这里ep并行，默认的叠加方式就是加权叠加，加权在各节点内先并行做，并且令未参与计算的部分会被设置成0，可以直接all reduce叠加。（这里的叠加方式就没得选了）
class EPMoE(torch.nn.Module):
    """
    MoE Expert Parallel Impl


    """

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        layer_id: int,
        params_dtype: Optional[torch.dtype] = None,
        renormalize: bool = True,
        use_grouped_topk: bool = False,
        num_expert_group: Optional[int] = None,
        num_fused_shared_experts: int = 0,
        topk_group: Optional[int] = None,
        quant_config: Optional[QuantizationConfig] = None,
        tp_size: Optional[int] = None,
        prefix: str = "",
        correction_bias: Optional[torch.Tensor] = None,
        custom_routing_function: Optional[Callable] = None,
        activation: str = "silu",
        routed_scaling_factor: Optional[float] = None,
        use_per_token_if_dynamic: bool = True,
    ):
        super().__init__()

        if params_dtype is None:
            params_dtype = torch.get_default_dtype()

        self.tp_size = (
            tp_size if tp_size is not None else get_tensor_model_parallel_world_size()
        )
        # 用了tp_rank代替ep_rank的意思。
        self.tp_rank = get_tensor_model_parallel_rank()

        self.layer_id = layer_id
        self.num_experts = num_experts
        assert self.num_experts % self.tp_size == 0
        assert (
            num_fused_shared_experts == 0
        ), "num_fused_shared_experts is not supported in EP"
        self.num_experts_per_partition = self.num_experts // self.tp_size
        # 标记该节点负责的专家的id号范围
        self.start_expert_id = self.tp_rank * self.num_experts_per_partition
        self.end_expert_id = self.start_expert_id + self.num_experts_per_partition - 1

        self.top_k = top_k
        self.intermediate_size = intermediate_size
        self.renormalize = renormalize
        self.use_grouped_topk = use_grouped_topk
        if self.use_grouped_topk:
            assert num_expert_group is not None and topk_group is not None
        self.num_expert_group = num_expert_group
        self.topk_group = topk_group
        self.correction_bias = correction_bias
        self.custom_routing_function = custom_routing_function
        self.activation = activation
        self.routed_scaling_factor = routed_scaling_factor
        self.use_per_token_if_dynamic = use_per_token_if_dynamic

        if quant_config is None:
            self.quant_method: Optional[QuantizeMethodBase] = UnquantizedEPMoEMethod()
            self.use_fp8_w8a8 = False
            self.use_block_quant = False
            self.block_shape = None
            self.activation_scheme = None
        else:
            # <NT> 目前只支持e4m3的fp8_w8a8。不支持e5m2，因为官方训练使用的是 E4M3 格式以获得更高的精度。
            # 而不是前向 E4M3，反向 E5M2。认为这种方法的可行性归功于细粒度量化策略，即 tile 和 block-wise scale。
            # 通过对更小组的元素进行操作，有效地在分组元素之间共享指数位，缓解了动态范围有限的影响。
            self.quant_method: Optional[QuantizeMethodBase] = Fp8EPMoEMethod(
                quant_config
            )
            self.use_fp8_w8a8 = True
            self.use_block_quant = getattr(self.quant_method, "block_quant", False)
            self.block_shape = (
                self.quant_method.quant_config.weight_block_size
                if self.use_block_quant
                else None
            )
            self.fp8_dtype = torch.float8_e4m3fn
            self.activation_scheme = quant_config.activation_scheme

        # <NT> 用选中的方法申请空间和指定权重加载方式。
        # 申请的空间包括 w13_weight, w13_weight_scale, w13_input_scale,
        #               w2_weight,  w2_weight_scale,  w2_input_scale
        self.quant_method.create_weights(
            layer=self,
            num_experts_per_partition=self.num_experts_per_partition,
            hidden_size=hidden_size,
            intermediate_size=self.intermediate_size,
            params_dtype=params_dtype,
            weight_loader=self.weight_loader,
        )

        self.grouped_gemm_runner = None

    def forward(self, hidden_states: torch.Tensor, router_logits: torch.Tensor):
        hidden_states_shape = hidden_states.shape
        hidden_states_dtype = hidden_states.dtype
        hidden_states_device = hidden_states.device

        assert self.quant_method is not None

        if self.grouped_gemm_runner is None:
            self.grouped_gemm_runner = GroupedGemmRunner(
                hidden_states.device,
                use_flashinfer=False,  # TODO: use flashinfer
                use_per_token_if_dynamic=self.use_per_token_if_dynamic,
            )
        
        # <NT> router_logits 是门控网路的输出，筛选出得到高分的专家网络, 
        # 即每个token要使用的topk个专家对应的权重和id号
        topk_weights, topk_ids = select_experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
            top_k=self.top_k,
            use_grouped_topk=self.use_grouped_topk,
            renormalize=self.renormalize,
            topk_group=self.topk_group,
            num_expert_group=self.num_expert_group,
            correction_bias=self.correction_bias,
            custom_routing_function=self.custom_routing_function,
            routed_scaling_factor=self.routed_scaling_factor,
            expert_location_dispatch_info=ExpertLocationDispatchInfo.init_new(
                layer_id=self.layer_id,
            ),
        )

        # <NT> 预处理topk id, 获取重排序信息
        reorder_topk_ids, src2dst, seg_indptr = run_moe_ep_preproess(
            topk_ids, self.num_experts
        )

        # <NT> 准备gate_up_proj的输入.
        gateup_input = torch.empty(
            (int(hidden_states.shape[0] * self.top_k), hidden_states.shape[1]),
            device=hidden_states.device,
            dtype=(
                self.fp8_dtype
                if (self.use_fp8_w8a8 and not self.use_block_quant)
                else hidden_states.dtype
            ),
        )
        # <NT> deepseek权重量化只支持静态，激活值量化支持静态和动态的。
        # 如激活值采用动态, 需要每次推理都计算一次input的scale值。
        if self.activation_scheme == "dynamic" and not self.use_block_quant:
            if self.use_per_token_if_dynamic:
                max_value = torch.max(hidden_states, dim=1).values.to(torch.float32)
                self.w13_input_scale = max_value / torch.finfo(self.fp8_dtype).max
            else:
                max_value = (
                    torch.max(hidden_states)
                    .repeat(self.num_experts_per_partition)
                    .to(torch.float32)
                )
                self.w13_input_scale = max_value / torch.finfo(self.fp8_dtype).max

        # <NT> 预重排序, 重新排列输入数据,将相同专家的数据分组在一起以便后续批量计算
        # hidden_states是输入, gateup_input是输出, gateup_input会充当gate_up_proj的输入.
        # PreReorder
        pre_reorder_triton_kernel[(hidden_states.shape[0],)](
            hidden_states,
            gateup_input,
            src2dst,
            topk_ids,
            self.w13_input_scale,
            self.start_expert_id,
            self.end_expert_id,
            self.top_k,
            hidden_states.shape[1],
            BLOCK_SIZE=512,
            use_per_token_if_dynamic=self.use_per_token_if_dynamic,
        )
        dispose_tensor(hidden_states)

        if (
            self.activation_scheme == "dynamic"
            and not self.use_block_quant
            and self.use_per_token_if_dynamic
        ):
            scale = torch.empty(
                hidden_states_shape[0] * self.top_k,
                device=hidden_states_device,
                dtype=torch.float32,
            )
            scale[src2dst] = (
                self.w13_input_scale.unsqueeze(1)
                .expand(hidden_states_shape[0], self.top_k)
                .reshape(-1)
            )
            self.w13_input_scale = scale

        # <NT> 获取当前rank的分段指针和权重索引
        seg_indptr_cur_rank = seg_indptr[self.start_expert_id : self.end_expert_id + 2]
        weight_indices_cur_rank = torch.arange(
            0,
            self.num_experts_per_partition,
            device=hidden_states_device,
            dtype=torch.int64,
        )
        # <NT> 分组矩阵乘，计算第一个线性层gate_up_proj (gate和up二合一)
        # GroupGemm-0
        gateup_output = self.grouped_gemm_runner(
            a=gateup_input,
            b=self.w13_weight,
            c=None,
            c_dtype=hidden_states_dtype,
            batch_size=self.num_experts_per_partition,
            weight_column_major=True,
            seg_indptr=seg_indptr_cur_rank,
            weight_indices=weight_indices_cur_rank,
            use_fp8_w8a8=self.use_fp8_w8a8,
            scale_a=self.w13_input_scale,
            scale_b=(
                self.w13_weight_scale_inv
                if self.use_block_quant
                else self.w13_weight_scale
            ),
            block_shape=self.block_shape,
        )
        del gateup_input

        # Act
        if self.activation_scheme == "dynamic" and not self.use_block_quant:
            self.w2_input_scale = None
            down_input = torch.empty(
                gateup_output.shape[0],
                gateup_output.shape[1] // 2,
                device=gateup_output.device,
                dtype=hidden_states_dtype,
            )
        else:
            down_input = torch.empty(
                gateup_output.shape[0],
                gateup_output.shape[1] // 2,
                device=gateup_output.device,
                dtype=(
                    self.fp8_dtype
                    if (self.use_fp8_w8a8 and not self.use_block_quant)
                    else hidden_states_dtype
                ),
            )

        if self.activation == "silu":
            silu_and_mul_triton_kernel[(gateup_output.shape[0],)](
                gateup_output,
                down_input,
                gateup_output.shape[1],
                reorder_topk_ids,
                self.w2_input_scale,
                self.start_expert_id,
                self.end_expert_id,
                BLOCK_SIZE=512,
            )
        elif self.activation == "gelu":
            gelu_and_mul_triton_kernel[(gateup_output.shape[0],)](
                gateup_output,
                down_input,
                gateup_output.shape[1],
                reorder_topk_ids,
                self.w2_input_scale,
                self.start_expert_id,
                self.end_expert_id,
                BLOCK_SIZE=512,
            )
        else:
            raise ValueError(f"Unsupported activation: {self.activation=}")
        del gateup_output

        if self.activation_scheme == "dynamic" and not self.use_block_quant:
            if self.use_per_token_if_dynamic:
                down_input, self.w2_input_scale = sglang_per_token_quant_fp8(down_input)
            else:
                self.w2_input_scale = torch.ones(
                    self.num_experts_per_partition,
                    dtype=torch.float32,
                    device=hidden_states_device,
                )

		# <NT> 分组矩阵乘，计算第二个线性层down_proj
        # GroupGemm-1
        down_output = torch.empty(
            down_input.shape[0],
            self.w2_weight.shape[1],
            device=hidden_states_device,
            dtype=hidden_states_dtype,
        )
        down_output = self.grouped_gemm_runner(
            a=down_input,
            b=self.w2_weight,
            c=down_output,
            batch_size=self.num_experts_per_partition,
            weight_column_major=True,
            seg_indptr=seg_indptr_cur_rank,
            weight_indices=weight_indices_cur_rank,
            use_fp8_w8a8=self.use_fp8_w8a8,
            scale_a=self.w2_input_scale,
            scale_b=(
                self.w2_weight_scale_inv
                if self.use_block_quant
                else self.w2_weight_scale
            ),
            block_shape=self.block_shape,
        )
        del down_input

		# <NT> 后重排序, 将各个专家的输出按原始token顺序重组,并根据专家权重进行加权组合得到最终输出
        # 输入down_output, 输出output
        # PostReorder
        output = torch.empty(
            hidden_states_shape, dtype=hidden_states_dtype, device=hidden_states_device
        )
        post_reorder_triton_kernel[(hidden_states_shape[0],)](
            down_output,
            output,
            src2dst,
            topk_ids,
            topk_weights,
            self.start_expert_id,
            self.end_expert_id,
            self.top_k,
            hidden_states_shape[1],
            BLOCK_SIZE=512,
        )
        return output

    # <NT> 生成参数映射表，是一个元组, 里面指定了(param_name, weight_name, expert_id, shard_id)，会被充当weight_loader函数的参数
    # make_expert_params_mapping在models/DeepseekV2ForCausalLM.load_weights中调用，喂入到weight_loader中。
    # ckpt_gate_proj_name: checkpoint中gate投影层的名称, 对应w1
    # ckpt_down_proj_name: checkpoint中down投影层的名称, 对应w2
    # ckpt_up_proj_name: checkpoint中up投影层的名称, 对应w3
    @classmethod
    def make_expert_params_mapping(
        cls,
        ckpt_gate_proj_name: str,
        ckpt_down_proj_name: str,
        ckpt_up_proj_name: str,
        num_experts: int,
    ) -> List[Tuple[str, str, int, str]]:
        return [
            # (param_name, weight_name, expert_id, shard_id)
            (
                (
                    "experts.w13_"
                    if weight_name in [ckpt_gate_proj_name, ckpt_up_proj_name]
                    else "experts.w2_"
                ),
                f"experts.{expert_id}.{weight_name}.",
                expert_id,
                shard_id,
            )
            for expert_id in range(num_experts)
            for shard_id, weight_name in [
                ("w1", ckpt_gate_proj_name),
                ("w2", ckpt_down_proj_name),
                ("w3", ckpt_up_proj_name),
            ]
        ]

    def weight_loader(
        self,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
        expert_id: int,
    ) -> None:
        physical_expert_ids = (
            get_global_expert_location_metadata().logical_to_all_physical(
                self.layer_id, expert_id
            )
        )
        for physical_expert_id in physical_expert_ids:
            self._weight_loader_physical(
                param=param,
                loaded_weight=loaded_weight,
                weight_name=weight_name,
                shard_id=shard_id,
                expert_id=physical_expert_id,
            )

    def _weight_loader_physical(
        self,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
        expert_id: int,
    ) -> None:
        # 约束只加载当前节点所负责的专家网络
        if expert_id < self.start_expert_id or expert_id > self.end_expert_id:
            return
        # 调整专家网络的id号，从全局的id调整为自己的局部id, 使其从0递增。
        expert_id = expert_id - self.start_expert_id

        # 权重名字必须是w1/w2/w3.
        if shard_id not in ("w1", "w2", "w3"):
            raise ValueError(
                f"shard_id must be ['w1','w2','w3'] but " f"got {shard_id}."
            )

        # Special case for fp8 scales.
        if "scale" in weight_name:
            self._load_fp8_scale(
                param.data,
                loaded_weight,
                weight_name,
                shard_id,
                expert_id,
            )
            return

		# 将权重对应写入即可
        if shard_id == "w2":
            param.data[expert_id] = loaded_weight
        elif shard_id == "w1":
            param.data[expert_id][: self.intermediate_size, :] = loaded_weight
        elif shard_id == "w3":
            param.data[expert_id][self.intermediate_size :, :] = loaded_weight
        else:
            raise ValueError(f"Expected shard_id w1,w2 or w3 but got {shard_id}")

    def _load_fp8_scale(
        self,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
        expert_id: int,
    ) -> None:
        param_data = param.data

        # Input scales can be loaded directly and should be equal.
        if "input_scale" in weight_name:
            if (
                (shard_id == "w1" or shard_id == "w3")
                and param_data[expert_id] != 1
                and (param_data[expert_id] - loaded_weight).abs() > 1e-5
            ):
                raise ValueError(
                    "input_scales of w1 and w3 of a layer "
                    f"must be equal. But got {param_data[expert_id]} "
                    f"vs. {loaded_weight}"
                )
            param_data[expert_id] = loaded_weight
        # Weight scales
        elif "weight_scale" in weight_name:
            if self.use_block_quant:
                block_n, block_k = self.block_shape[0], self.block_shape[1]
                if shard_id == "w1":
                    param_data[expert_id][
                        : (self.intermediate_size + block_n - 1) // block_n, :
                    ] = loaded_weight
                elif shard_id == "w3":
                    param_data[expert_id][
                        (self.intermediate_size + block_n - 1) // block_n :, :
                    ] = loaded_weight
                else:  # w2
                    param_data[expert_id] = loaded_weight
            # If we are in merged column case (gate_up_proj)
            else:
                if shard_id in ("w1", "w3"):
                    # We have to keep the weight scales of w1 and w3 because
                    # we need to re-quantize w1/w3 weights after weight loading.
                    idx = 0 if shard_id == "w1" else 1
                    param_data[expert_id][idx] = loaded_weight

                # If we are in the row parallel case (down_proj)
                else:
                    param_data[expert_id] = loaded_weight


class UnquantizedEPMoEMethod(FusedMoEMethodBase, CustomOp):

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts_per_partition: int,
        hidden_size: int,
        intermediate_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        # <NT> (out_size, in_size) = (2*intermediate_size, hidden_size)
        # 正常矩阵乘的B矩阵应该是要(in_size，out_size)，为行优先。所以目前的权重格式需要按列优先来充当B矩阵。
        # Fused gate_up_proj (column parallel)
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts_per_partition,
                2 * intermediate_size,
                hidden_size,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        # down_proj (row parallel)
        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts_per_partition,
                hidden_size,
                intermediate_size,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # scale
        layer.register_parameter("w13_input_scale", None)
        layer.register_parameter("w13_weight_scale", None)

        ones_tensor = torch.ones(num_experts_per_partition, dtype=torch.float32)

        w2_input_scale = torch.nn.Parameter(
            ones_tensor,
            requires_grad=False,
        )
        layer.register_parameter("w2_input_scale", w2_input_scale)
        set_weight_attrs(w2_input_scale, extra_weight_attrs)

        w2_weight_scale = torch.nn.Parameter(
            ones_tensor,
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        custom_routing_function: Optional[Callable] = None,
    ) -> torch.Tensor:
        raise NotImplementedError


class Fp8EPMoEMethod(Fp8MoEMethod):
    """MoE method for FP8.
    <NT> 支持静态量化的weight scale和动态/静态量化的activation scale.
    Supports loading FP8 checkpoints with static weight scale and
    dynamic/static activation scale.

    Args:
        quant_config: The quantization config.
    """

    def __init__(self, quant_config: Fp8Config):
        self.quant_config = quant_config
        self.block_quant = self.quant_config.weight_block_size is not None

    # <NT> **extra_weight_attrs为可变关键字参数，允许函数接受任意数量的关键字参数，这些参数会被收集到一个字典中，字典的键是参数名，值是参数值。
    # 上面EPMoE的init中调用create_weights时，将weight_loader=self.weight_loader，即会被传到extra_weight_attrs里，然后在set_weight_attrs中使用。
    # 将权重加载函数作为对应层的属性记录到模型中，以便使用self.named_parameters可以访问到。
    # 所以这里create_weights并未调用到self.weight_loader，不会进行实际的权重加载操作，但会把权重加载函数注册进去。
    # 实际的权重加载操作在models/DeepseekV2ForCausalLM.load_weights中，通过self.named_parameters把注册好的权重加载拿出来，进行实际的加载操作。
    def create_weights(
        self,
        layer: Module,
        num_experts_per_partition: int,
        hidden_size: int,
        intermediate_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):

        if self.quant_config.is_checkpoint_fp8_serialized:
            params_dtype = torch.float8_e4m3fn

        tp_size = get_tensor_model_parallel_world_size()
        if self.block_quant:
            block_n, block_k = (
                self.quant_config.weight_block_size[0],
                self.quant_config.weight_block_size[1],
            )
            # NOTE(HandH1998): To ensure proper alignment of the block-wise quantization scales, the output_size of the weights for both the gate and up layers must be divisible by block_n.
            # Required by column parallel or enabling merged weights
            if intermediate_size % block_n != 0:
                raise ValueError(
                    f"The output_size of gate's and up's weight = "
                    f"{intermediate_size} is not divisible by "
                    f"weight quantization block_n = {block_n}."
                )
            if tp_size > 1:
                # Required by row parallel
                if intermediate_size % block_k != 0:
                    raise ValueError(
                        f"The input_size of down's weight = "
                        f"{intermediate_size} is not divisible by "
                        f"weight quantization block_k = {block_k}."
                    )

		# <NT> 权重分为w1/w3和w2, w1和w3对应gate_proj+up_proj=gate_up_proj,w2对应down_proj，up_proj->act->down_proj共同定义了专家网络的结构,
        # 
        # 如下，w1和w3用一个三维矩阵放在一起[num_experts_per_partition, 2 * intermediate_size, hidden_size] => *size
        # torch.empty的默认layout是torch.strided，默认情况下，最后一维连续排布，从后往前step。
        # w1和w3的维度都是[num_experts_per_partition, out=intermediate_size, in=hidden_size], w1和w3会合并作为一个线性层的权重一起计算。
        # w2的维度是      [num_experts_per_partition, out=hidden_size, in=intermediate_size]
        # 因为是EP专家并行，每个节点只需要加载自己所负责专家网络即可，即num_experts_per_partition，表示每个part负责的专家数量。
        # 
        # 对于nn.Linear中权重维度是[out_features, in_features], 
        # w13_weight对应输入是hidden_size, 输出size是2 * intermediate_size, intermediate_size大于hidden_size，升维度。
        # w2_weight对应输入是intermediate_size，输出是hidden_size，降维。up_proj后会做SiluAndMul，会将2 * intermediate_size合成intermediate_size，参考moe_forward_native实现。
        #
        # WEIGHTS
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts_per_partition,
                2 * intermediate_size,
                hidden_size,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts_per_partition,
                hidden_size,
                intermediate_size,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # WEIGHT_SCALES
        # <NT> weight_scale是一个专家一个，w1w2w3都各一份。
        #      input_scale也是一个专家一个，w1和w3的是相同的输入，公用一个，w2自己一个。
        if self.block_quant:
            w13_weight_scale = torch.nn.Parameter(
                torch.ones(
                    num_experts_per_partition,
                    2 * ((intermediate_size + block_n - 1) // block_n),
                    (hidden_size + block_k - 1) // block_k,
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
            w2_weight_scale = torch.nn.Parameter(
                torch.ones(
                    num_experts_per_partition,
                    (hidden_size + block_n - 1) // block_n,
                    (intermediate_size + block_k - 1) // block_k,
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w13_weight_scale_inv", w13_weight_scale)
            layer.register_parameter("w2_weight_scale_inv", w2_weight_scale)
            assert self.quant_config.activation_scheme == "dynamic"
        else:
            # WEIGHT_SCALES
            # Allocate 2 scales for w1 and w3 respectively.
            w13_weight_scale = torch.nn.Parameter(
                torch.ones(num_experts_per_partition, 2, dtype=torch.float32),
                requires_grad=False,
            )
            layer.register_parameter("w13_weight_scale", w13_weight_scale)

            w2_weight_scale = torch.nn.Parameter(
                torch.ones(num_experts_per_partition, dtype=torch.float32),
                requires_grad=False,
            )
            layer.register_parameter("w2_weight_scale", w2_weight_scale)
        # Add the quantization method used (per tensor/grouped/channel)
        # to ensure the weight scales are loaded in properly
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.BLOCK.value}
            if self.block_quant
            else {"quant_method": FusedMoeWeightScaleSupported.TENSOR.value}
        )
        # If loading fp8 checkpoint, pass the weight loaders.
        # If loading an fp16 checkpoint, do not (we will quantize in
        #   process_weights_after_loading()
        if self.quant_config.is_checkpoint_fp8_serialized:
            set_weight_attrs(w13_weight_scale, extra_weight_attrs)
            set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        # INPUT_SCALES
        if self.quant_config.activation_scheme == "static":
            if not self.quant_config.is_checkpoint_fp8_serialized:
                raise ValueError(
                    "Found static activation scheme for checkpoint that "
                    "was not serialized fp8."
                )

            w13_input_scale = torch.nn.Parameter(
                torch.ones(num_experts_per_partition, dtype=torch.float32),
                requires_grad=False,
            )
            layer.register_parameter("w13_input_scale", w13_input_scale)
            set_weight_attrs(w13_input_scale, extra_weight_attrs)

            w2_input_scale = torch.nn.Parameter(
                torch.ones(num_experts_per_partition, dtype=torch.float32),
                requires_grad=False,
            )
            layer.register_parameter("w2_input_scale", w2_input_scale)
            set_weight_attrs(w2_input_scale, extra_weight_attrs)

        else:
            layer.w13_input_scale = None
            layer.w2_input_scale = None

    def process_weights_after_loading(self, layer: Module) -> None:

        # If checkpoint is fp16, quantize in place.
        if not self.quant_config.is_checkpoint_fp8_serialized:
            # If rocm, use float8_e4m3fnuz as dtype
            fp8_dtype = torch.float8_e4m3fnuz if _is_hip else torch.float8_e4m3fn
            w13_weight = torch.empty_like(layer.w13_weight.data, dtype=fp8_dtype)
            w2_weight = torch.empty_like(layer.w2_weight.data, dtype=fp8_dtype)

            layer.w13_weight_scale = torch.nn.Parameter(
                torch.ones(
                    layer.num_experts_per_partition,
                    dtype=torch.float32,
                    device=w13_weight.device,
                ),
                requires_grad=False,
            )

            for expert in range(layer.num_experts_per_partition):
                w13_weight[expert, :, :], layer.w13_weight_scale[expert] = (
                    scaled_fp8_quant(layer.w13_weight.data[expert, :, :])
                )
                w2_weight[expert, :, :], layer.w2_weight_scale[expert] = (
                    scaled_fp8_quant(layer.w2_weight.data[expert, :, :])
                )
            layer.w13_weight = torch.nn.Parameter(w13_weight, requires_grad=False)
            layer.w2_weight = torch.nn.Parameter(w2_weight, requires_grad=False)
            return

        # If checkpoint is fp8, we need to handle that the
        # MoE kernels require single activation scale and single weight
        # scale for w13 per expert.
        else:
            if self.quant_config.activation_scheme == "static":
                if layer.w13_input_scale is None or layer.w2_input_scale is None:
                    raise ValueError(
                        "QuantConfig has static quantization, but found "
                        "activation scales are None."
                    )
                layer.w13_weight_scale = torch.nn.Parameter(
                    torch.max(layer.w13_weight_scale, dim=1).values,
                    requires_grad=False,
                )
            return

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        custom_routing_function: Optional[Callable] = None,
    ) -> torch.Tensor:
        raise NotImplementedError


class DeepEPMoE(EPMoE):
    """
    MoE Expert Parallel Impl based on DeepEP (https://github.com/deepseek-ai/DeepEP/tree/main)
    """

    _has_printed = False

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        layer_id: int,
        params_dtype: Optional[torch.dtype] = None,
        renormalize: bool = True,
        use_grouped_topk: bool = False,
        num_expert_group: Optional[int] = None,
        num_fused_shared_experts: int = 0,
        topk_group: Optional[int] = None,
        quant_config: Optional[QuantizationConfig] = None,
        tp_size: Optional[int] = None,
        prefix: str = "",
        correction_bias: Optional[torch.Tensor] = None,
        custom_routing_function: Optional[Callable] = None,
        activation: str = "silu",
        routed_scaling_factor: Optional[float] = None,
        deepep_mode: DeepEPMode = DeepEPMode.auto,
    ):
        super().__init__(
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            layer_id=layer_id,
            params_dtype=params_dtype,
            renormalize=renormalize,
            use_grouped_topk=use_grouped_topk,
            num_expert_group=num_expert_group,
            num_fused_shared_experts=num_fused_shared_experts,
            topk_group=topk_group,
            quant_config=quant_config,
            tp_size=tp_size,
            prefix=prefix,
            correction_bias=correction_bias,
            custom_routing_function=custom_routing_function,
            activation=activation,
            routed_scaling_factor=routed_scaling_factor,
        )
        self.deepep_mode = deepep_mode
        if self.deepep_mode.enable_low_latency():
            assert use_deep_gemm, f"DeepEP {self.deepep_mode} mode requires deep_gemm"
        self.w13_weight_fp8 = (
            self.w13_weight,
            (
                self.w13_weight_scale_inv
                if self.use_block_quant
                else self.w13_weight_scale
            ),
        )
        self.w2_weight_fp8 = (
            self.w2_weight,
            self.w2_weight_scale_inv if self.use_block_quant else self.w2_weight_scale,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        reorder_topk_ids: torch.Tensor,
        seg_indptr: torch.Tensor,
        masked_m: torch.Tensor,
        expected_m: int,
        num_recv_tokens_per_expert: List[int],
        forward_mode: ForwardMode,
    ):
        resolved_deepep_mode = self.deepep_mode.resolve(forward_mode)
        if resolved_deepep_mode == DeepEPMode.normal:
            if _ENABLE_JIT_DEEPGEMM:
                return self.forward_deepgemm_contiguous(
                    hidden_states, topk_idx, topk_weights, num_recv_tokens_per_expert
                )
            else:
                return self.forward_normal(hidden_states, reorder_topk_ids, seg_indptr)
        elif resolved_deepep_mode == DeepEPMode.low_latency:
            return self.forward_deepgemm_masked(hidden_states, masked_m, expected_m)
        else:
            raise ValueError(f"Invalid deepep_mode: {self.deepep_mode}")

    def forward_normal(
        self,
        hidden_states: torch.Tensor,
        reorder_topk_ids: torch.Tensor,
        seg_indptr: torch.Tensor,
    ):
        hidden_states_dtype = hidden_states.dtype
        hidden_states_device = hidden_states.device

        assert self.quant_method is not None
        assert self.activation == "silu"
        if self.grouped_gemm_runner is None:
            self.grouped_gemm_runner = GroupedGemmRunner(
                hidden_states.device, use_flashinfer=False  # TODO: use flashinfer
            )

        if self.activation_scheme == "dynamic" and not self.use_block_quant:
            max_value = (
                torch.max(hidden_states)
                .repeat(self.num_experts_per_partition)
                .to(torch.float32)
            )
            self.w13_input_scale = max_value / torch.finfo(self.fp8_dtype).max
        weight_indices_cur_rank = torch.arange(
            0,
            self.num_experts_per_partition,
            device=hidden_states.device,
            dtype=torch.int64,
        )

        # GroupGemm-0
        if hidden_states.shape[0] > 0:
            gateup_output = self.grouped_gemm_runner(
                a=hidden_states,
                b=self.w13_weight,
                c=None,
                c_dtype=hidden_states.dtype,
                batch_size=self.num_experts_per_partition,
                weight_column_major=True,
                seg_indptr=seg_indptr,
                weight_indices=weight_indices_cur_rank,
                use_fp8_w8a8=self.use_fp8_w8a8,
                scale_a=self.w13_input_scale,
                scale_b=(
                    self.w13_weight_scale_inv
                    if self.use_block_quant
                    else self.w13_weight_scale
                ),
                block_shape=self.block_shape,
            )
        else:
            gateup_output = torch.empty(
                hidden_states.shape[0],
                self.w13_weight.shape[1],
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )

        # Act
        down_input = torch.empty(
            gateup_output.shape[0],
            gateup_output.shape[1] // 2,
            device=gateup_output.device,
            dtype=(
                self.fp8_dtype
                if (self.use_fp8_w8a8 and not self.use_block_quant)
                else hidden_states_dtype
            ),
        )
        if self.w2_input_scale is None and not self.use_block_quant:
            self.w2_input_scale = torch.ones(
                self.num_experts_per_partition,
                dtype=torch.float32,
                device=hidden_states_device,
            )

        if self.activation == "silu":
            silu_and_mul_triton_kernel[(gateup_output.shape[0],)](
                gateup_output,
                down_input,
                gateup_output.shape[1],
                reorder_topk_ids,
                self.w2_input_scale,
                0,
                self.num_experts_per_partition - 1,
                BLOCK_SIZE=512,
            )
        else:
            raise ValueError(f"Unsupported activation: {self.activation=}")

        del gateup_output

        # GroupGemm-1
        down_output = torch.empty(
            down_input.shape[0],
            self.w2_weight.shape[1],
            device=hidden_states_device,
            dtype=hidden_states_dtype,
        )
        if down_input.shape[0] > 0:
            down_output = self.grouped_gemm_runner(
                a=down_input,
                b=self.w2_weight,
                c=down_output,
                batch_size=self.num_experts_per_partition,
                weight_column_major=True,
                seg_indptr=seg_indptr,
                weight_indices=weight_indices_cur_rank,
                use_fp8_w8a8=self.use_fp8_w8a8,
                scale_a=self.w2_input_scale,
                scale_b=(
                    self.w2_weight_scale_inv
                    if self.use_block_quant
                    else self.w2_weight_scale
                ),
                block_shape=self.block_shape,
            )
        return down_output

    def forward_deepgemm_contiguous(
        self,
        hidden_states_fp8: Tuple[torch.Tensor, torch.Tensor],
        topk_idx,
        topk_weights,
        num_recv_tokens_per_expert: List[int],
    ):
        hidden_states_fp8, hidden_states_scale = hidden_states_fp8
        assert self.quant_method is not None
        assert self.activation == "silu"
        if num_recv_tokens_per_expert is None:
            return hidden_states_fp8.bfloat16()
        all_tokens = sum(num_recv_tokens_per_expert)
        if all_tokens <= 0:
            return hidden_states_fp8.bfloat16()
        M, K = hidden_states_fp8.size()
        N = self.w13_weight.size(1)
        scale_block_size = 128

        hidden_states_fp8_shape = hidden_states_fp8.shape
        hidden_states_fp8_device = hidden_states_fp8.device
        hidden_states_fp8_dtype = hidden_states_fp8.dtype

        input_tensor = [
            torch.empty(
                (all_tokens, K),
                device=hidden_states_fp8.device,
                dtype=hidden_states_fp8.dtype,
            ),
            torch.empty(
                (all_tokens, K // 128),
                device=hidden_states_fp8.device,
                dtype=torch.float32,
            ),
        ]
        m_indices = torch.empty(
            all_tokens, device=hidden_states_fp8.device, dtype=torch.int32
        )
        output_index = torch.empty_like(topk_idx)

        num_recv_tokens_per_expert_gpu = torch.tensor(
            num_recv_tokens_per_expert,
            dtype=torch.int32,
            pin_memory=True,
            device="cpu",
        ).cuda(non_blocking=True)
        expert_start_loc = torch.empty_like(num_recv_tokens_per_expert_gpu)

        ep_scatter(
            hidden_states_fp8,
            hidden_states_scale,
            topk_idx,
            num_recv_tokens_per_expert_gpu,
            expert_start_loc,
            input_tensor[0],
            input_tensor[1],
            m_indices,
            output_index,
        )
        dispose_tensor(hidden_states_fp8)

        gateup_output = torch.empty(
            (all_tokens, N),
            device=hidden_states_fp8_device,
            dtype=torch.bfloat16,
        )
        input_tensor[1] = tma_align_input_scale(input_tensor[1])
        m_grouped_gemm_fp8_fp8_bf16_nt_contiguous(
            input_tensor, self.w13_weight_fp8, gateup_output, m_indices
        )
        del input_tensor
        down_input = torch.empty(
            (
                all_tokens,
                N // 2,
            ),
            device=gateup_output.device,
            dtype=torch.bfloat16,
        )
        silu_and_mul(gateup_output.view(-1, N), down_input)
        del gateup_output
        down_output = torch.empty(
            (all_tokens, K),
            device=hidden_states_fp8_device,
            dtype=torch.bfloat16,
        )
        down_input_fp8, down_input_scale = sglang_per_token_group_quant_fp8(
            down_input, scale_block_size
        )
        del down_input
        down_input_scale = tma_align_input_scale(down_input_scale)
        m_grouped_gemm_fp8_fp8_bf16_nt_contiguous(
            (down_input_fp8, down_input_scale),
            self.w2_weight_fp8,
            down_output,
            m_indices,
        )
        del down_input_fp8, down_input_scale

        gather_out = torch.empty(
            hidden_states_fp8_shape,
            device=hidden_states_fp8_device,
            dtype=torch.bfloat16,
        )
        ep_gather(down_output, topk_idx, topk_weights, output_index, gather_out)

        return gather_out

    def forward_deepgemm_masked(
        self,
        hidden_states_fp8: Tuple[torch.Tensor, torch.Tensor],
        masked_m: torch.Tensor,
        expected_m: int,
    ):
        assert self.quant_method is not None
        assert self.activation == "silu"

        # GroupGemm-0
        num_groups, m, k = hidden_states_fp8[0].size()
        n = self.w13_weight.size(1)
        expected_m = min(expected_m, m)
        gateup_output = torch.empty(
            (num_groups, m, n), device=hidden_states_fp8[0].device, dtype=torch.bfloat16
        )
        m_grouped_gemm_fp8_fp8_bf16_nt_masked(
            hidden_states_fp8, self.w13_weight_fp8, gateup_output, masked_m, expected_m
        )
        dispose_tensor(hidden_states_fp8[0])

        # Act
        down_input = torch.empty(
            (
                gateup_output.shape[0],
                gateup_output.shape[1],
                gateup_output.shape[2] // 2,
            ),
            device=gateup_output.device,
            dtype=self.fp8_dtype,
        )
        scale_block_size = 128
        down_input_scale = torch.empty(
            (
                gateup_output.shape[0],
                gateup_output.shape[1],
                gateup_output.shape[2] // 2 // scale_block_size,
            ),
            device=gateup_output.device,
            dtype=torch.float32,
        )
        silu_and_mul_masked_post_quant_fwd(
            gateup_output,
            down_input,
            down_input_scale,
            scale_block_size,
            masked_m,
        )
        del gateup_output

        # GroupGemm-1
        n = self.w2_weight.size(1)
        down_input_fp8 = (
            down_input,
            get_col_major_tma_aligned_tensor(down_input_scale),
        )
        down_output = torch.empty(
            (num_groups, m, n), device=down_input.device, dtype=torch.bfloat16
        )
        m_grouped_gemm_fp8_fp8_bf16_nt_masked(
            down_input_fp8, self.w2_weight_fp8, down_output, masked_m, expected_m
        )

        return down_output


def get_moe_impl_class():
    if global_server_args_dict["enable_deepep_moe"]:
        return DeepEPMoE
    if global_server_args_dict["enable_ep_moe"]:
        return EPMoE
    return FusedMoE
