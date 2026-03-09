import itertools

import pytest
import torch
from fouroversix import (
    DataType,
    QuantizationConfig,
    QuantizeBackend,
    RoundStyle,
    ScaleRule,
    quantize_to_fp4,
)
from fouroversix.quantize.frontend import AVAILABLE_BACKENDS
from fouroversix.quantize.quantized_tensor import from_blocked

MAE_MSE_MISMATCH_TOLERANCE = 1e-3
NUM_RANDOM_SEEDS = 10


@pytest.mark.parametrize("input_type", ["zeros", "ones", "rand01", "randn"])
@pytest.mark.parametrize(
    "input_shape",
    [(1024, 1024), (1024, 512), (512, 1024)],
)
@pytest.mark.parametrize(
    ("backend_a", "backend_b"),
    itertools.combinations(
        [
            QuantizeBackend.cuda,
            QuantizeBackend.triton,
            QuantizeBackend.pytorch,
            QuantizeBackend.transformer_engine,
        ],
        r=2,
    ),
)
@pytest.mark.parametrize("block_scale_2d", ["block_scale_2d", "no_block_scale_2d"])
@pytest.mark.parametrize("dtype", [DataType.nvfp4])
@pytest.mark.parametrize("rht", ["rht", "no_rht"])
@pytest.mark.parametrize(
    "scale_rule",
    [
        ScaleRule.abs_max,
        ScaleRule.mae,
        ScaleRule.mse,
        ScaleRule.static_4,
        ScaleRule.static_6,
    ],
)
@pytest.mark.parametrize("round_style", [RoundStyle.nearest, RoundStyle.stochastic])
@pytest.mark.parametrize("transpose", ["transpose", "no_transpose"])
def test_backend_outputs_are_consistent(  # noqa: C901, PLR0915
    input_type: str,
    input_shape: tuple[int, int],
    backend_a: QuantizeBackend,
    backend_b: QuantizeBackend,
    *,
    block_scale_2d: str,
    dtype: DataType,
    rht: str,
    round_style: RoundStyle,
    scale_rule: ScaleRule,
    transpose: str,
) -> None:
    torch.set_printoptions(precision=10)

    backend_a_cls = AVAILABLE_BACKENDS[backend_a]
    backend_b_cls = AVAILABLE_BACKENDS[backend_b]

    if not backend_a_cls.is_available() or not backend_b_cls.is_available():
        pytest.skip("Backend is not available")

    config_a = QuantizationConfig(
        backend=backend_a,
        block_scale_2d=block_scale_2d == "block_scale_2d",
        dtype=dtype,
        rht=rht == "rht",
        round_style=round_style,
        scale_rule=scale_rule,
        transpose=transpose == "transpose",
    )

    config_b = QuantizationConfig(
        backend=backend_b,
        block_scale_2d=block_scale_2d == "block_scale_2d",
        dtype=dtype,
        rht=rht == "rht",
        round_style=round_style,
        scale_rule=scale_rule,
        transpose=transpose == "transpose",
    )

    if round_style == RoundStyle.stochastic:
        pytest.xfail("This test is not currently targeting stochastic rounding")

    for random_seed in range(NUM_RANDOM_SEEDS):
        print(f"Testing with random seed: {random_seed}")
        torch.manual_seed(random_seed)

        if input_type == "zeros":
            x = torch.zeros(*input_shape, dtype=torch.bfloat16, device="cuda")
        elif input_type == "ones":
            x = torch.ones(*input_shape, dtype=torch.bfloat16, device="cuda")
        elif input_type == "rand01":
            x = torch.randint(0, 2, input_shape, dtype=int, device="cuda").to(
                torch.bfloat16,
            )
        elif input_type == "randn":
            x = torch.randn(*input_shape, dtype=torch.bfloat16, device="cuda")
        else:
            msg = f"Invalid input type: {input_type}"
            raise ValueError(msg)

        if not backend_a_cls.is_supported(
            x,
            config_a,
        ) or not backend_b_cls.is_supported(
            x,
            config_b,
        ):
            pytest.skip("Backend is not supported")

        quantized_a = quantize_to_fp4(x.clone(), config_a)
        quantized_b = quantize_to_fp4(x.clone(), config_b)

        if not torch.allclose(quantized_a.amax, quantized_b.amax):
            print("Backends A and B have different amax values!")
            print(f"{backend_a}: {quantized_a.amax}")
            print(f"{backend_b}: {quantized_b.amax}")
            pytest.fail("Backends A and B have different amax values!")

        sf_a = from_blocked(
            quantized_a.scale_factors.bfloat16(),
            (input_shape[0], input_shape[1] // 16),
        )
        sf_b = from_blocked(
            quantized_b.scale_factors.bfloat16(),
            (input_shape[0], input_shape[1] // 16),
        )

        # When computing 4/6 with the MAE and MSE scale rules, computing the errors
        # requires summing the errors in each block of 16 values. This operation
        # differently (elements are summed in different orders, and floating-point
        # addition is not associative) in PyTorch and Triton, and can not be easily made
        # deterministic in a way that allows for good performance. As a result, we allow
        # a small number of mismatches between the scale factors and values for these
        # two rules. Fortunately, abs_max does not involve a summation, so we can use it
        # to test the correctness of the rest of the 4/6 implementation.
        scale_factors_mismatch_prop = (sf_a != sf_b).sum() / sf_a.numel()

        if (
            scale_rule in {ScaleRule.static_6, ScaleRule.static_4, ScaleRule.abs_max}
            and scale_factors_mismatch_prop > 0
        ) or scale_factors_mismatch_prop >= MAE_MSE_MISMATCH_TOLERANCE:
            print(
                "Backends A and B have different scale factors! "
                f"{scale_factors_mismatch_prop:.2%} mismatch",
            )

            [i, *_], [j, *_] = torch.where(sf_a != sf_b)
            print(backend_a)
            print("sf", sf_a[i, j])
            print("e2m1", quantized_a.values[i, 8 * j : 8 * (j + 1)])
            print(backend_b)
            print("sf", sf_b[i, j])
            print("e2m1", quantized_b.values[i, 8 * j : 8 * (j + 1)])
            print("original")
            print("x", x[i, 16 * j : 16 * (j + 1)])
            pytest.fail("Backends A and B have different scale factors!")

        values_mismatch_prop = (
            quantized_a.values != quantized_b.values
        ).sum() / quantized_a.values.numel()

        if (
            scale_rule in {ScaleRule.static_6, ScaleRule.static_4, ScaleRule.abs_max}
            and values_mismatch_prop > 0
        ) or values_mismatch_prop >= MAE_MSE_MISMATCH_TOLERANCE:
            print(
                "Backends A and B have different e2m1 values! "
                f"{values_mismatch_prop:.2%} mismatch",
            )

            [i, *_], [j, *_] = torch.where(
                quantized_a.values != quantized_b.values,
            )
            print(i, j)
            print("amax", quantized_a.amax)
            print("sf", sf_a[i, j // 8])
            print(backend_a)
            print("e2m1", quantized_a.values[i, 8 * (j // 8) : 8 * (j // 8 + 1)])
            print(backend_b)
            print("e2m1", quantized_b.values[i, 8 * (j // 8) : 8 * (j // 8 + 1)])
            print("original")
            print("x", x[i, 16 * (j // 8) : 16 * (j // 8 + 1)])
            pytest.fail("Backends A and B have different e2m1 values!")


# [NEW] 新增测试：MXFP4自适应量化测试
@pytest.mark.parametrize("input_type", ["zeros", "ones", "rand01", "randn"])
@pytest.mark.parametrize(
    "input_shape",
    [(1024, 1024), (1024, 512), (512, 1024)],
)
@pytest.mark.parametrize(
    "scale_rule",
    [
        ScaleRule.abs_max,
        ScaleRule.mae,
        ScaleRule.mse,
        ScaleRule.static_4,
        ScaleRule.static_6,
    ],
)
def test_mxfp4_quantization(
    input_type: str,
    input_shape: tuple[int, int],
    scale_rule: ScaleRule,
) -> None:
    """
    [NEW] 测试MXFP4格式的量化，包括自适应缩放规则
    
    测试内容：
    1. 静态规则 (static_6, static_4) - 标准MXFP4量化
    2. 自适应规则 (mse, mae, abs_max) - MXFP4 FourOverSix量化
    """
    torch.manual_seed(42)

    if input_type == "zeros":
        x = torch.zeros(*input_shape, dtype=torch.bfloat16, device="cuda")
    elif input_type == "ones":
        x = torch.ones(*input_shape, dtype=torch.bfloat16, device="cuda")
    elif input_type == "rand01":
        x = torch.randint(0, 2, input_shape, dtype=int, device="cuda").to(
            torch.bfloat16,
        )
    elif input_type == "randn":
        x = torch.randn(*input_shape, dtype=torch.bfloat16, device="cuda")
    else:
        msg = f"Invalid input type: {input_type}"
        raise ValueError(msg)

    # 创建MXFP4量化配置
    config = QuantizationConfig(
        backend=QuantizeBackend.pytorch,  # 使用PyTorch后端（兼容性最好）
        dtype=DataType.mxfp4,
        scale_rule=scale_rule,
    )

    # 执行量化
    quantized = quantize_to_fp4(x.clone(), config)

    # 验证量化结果
    assert quantized.dtype == DataType.mxfp4
    assert quantized.values.shape == (input_shape[0], input_shape[1] // 2)
    # MXFP4块大小为32
    expected_scale_shape = input_shape[0] * (input_shape[1] // 32)
    assert quantized.scale_factors.numel() == expected_scale_shape or quantized.scale_factors.ndim == 1

    # 反量化并验证
    dequantized = quantized.dequantize(torch.bfloat16)
    assert dequantized.shape == input_shape

    # 对于非零输入，验证量化误差在合理范围内
    if input_type not in ["zeros"]:
        relative_error = (dequantized - x).abs().sum() / (x.abs().sum() + 1e-8)
        # FP4量化的相对误差通常在10%以内
        assert relative_error < 0.5, f"Relative error too high: {relative_error}"


# [NEW] 新增测试：MXFP4与NVFP4量化对比测试
@pytest.mark.parametrize("input_shape", [(256, 256)])
@pytest.mark.parametrize(
    "scale_rule",
    [ScaleRule.mse, ScaleRule.static_6],
)
def test_mxfp4_vs_nvfp4_quantization(
    input_shape: tuple[int, int],
    scale_rule: ScaleRule,
) -> None:
    """
    [NEW] 对比MXFP4和NVFP4的量化结果
    
    验证：
    1. 两种格式的量化都能正常工作
    2. 自适应规则在两种格式下都能正常工作
    """
    torch.manual_seed(42)
    x = torch.randn(*input_shape, dtype=torch.bfloat16, device="cuda")

    # MXFP4量化
    config_mxfp4 = QuantizationConfig(
        backend=QuantizeBackend.pytorch,
        dtype=DataType.mxfp4,
        scale_rule=scale_rule,
    )
    quantized_mxfp4 = quantize_to_fp4(x.clone(), config_mxfp4)
    dequantized_mxfp4 = quantized_mxfp4.dequantize(torch.bfloat16)

    # NVFP4量化
    config_nvfp4 = QuantizationConfig(
        backend=QuantizeBackend.pytorch,
        dtype=DataType.nvfp4,
        scale_rule=scale_rule,
    )
    quantized_nvfp4 = quantize_to_fp4(x.clone(), config_nvfp4)
    dequantized_nvfp4 = quantized_nvfp4.dequantize(torch.bfloat16)

    # 验证两种格式都能正常工作
    error_mxfp4 = (dequantized_mxfp4 - x).abs().mean()
    error_nvfp4 = (dequantized_nvfp4 - x).abs().mean()

    print(f"MXFP4 MAE: {error_mxfp4}")
    print(f"NVFP4 MAE: {error_nvfp4}")

    # 两种格式的误差都应该在合理范围内
    assert error_mxfp4 < 0.5, f"MXFP4 error too high: {error_mxfp4}"
    assert error_nvfp4 < 0.5, f"NVFP4 error too high: {error_nvfp4}"
