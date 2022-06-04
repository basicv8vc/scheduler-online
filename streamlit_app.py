import time
import re

import streamlit as st
import oneflow as flow

import numpy as np
import pandas as pd
import altair as alt
from altair import X, Y, Axis

ConstantLR_CODE = """oneflow.optim.lr_scheduler.ConstantLR(
                optimizer: Optimizer,
                factor: float = 1.0 / 3,
                total_iters: int = 5,
                last_step: int = -1,
                verbose: bool = False
                )"""

LinearLR_CODE = """oneflow.optim.lr_scheduler.LinearLR(
                optimizer: Optimizer,
                start_factor: float = 1.0 / 3,
                end_factor: float = 1.0,
                total_iters: int = 5,
                last_step: int = -1,
                verbose: bool = False,
                )"""
ExponentialLR_CODE = """oneflow.optim.lr_scheduler.ExponentialLR(
                optimizer: Optimizer,
                gamma: float,
                last_step: int = -1,
                verbose: bool = False,
                )"""

StepLR_CODE = """oneflow.optim.lr_scheduler.StepLR(
                optimizer: Optimizer,
                step_size: int,
                gamma: float = 0.1,
                last_step: int = -1,
                verbose: bool = False,
                )"""

MultiStepLR_CODE = """oneflow.optim.lr_scheduler.MultiStepLR(
                optimizer: Optimizer,
                milestones: list,
                gamma: float = 0.1,
                last_step: int = -1,
                verbose: bool = False,
                )"""

PolynomialLR_CODE = """oneflow.optim.lr_scheduler.PolynomialLR(
                optimizer,
                steps: int,
                end_learning_rate: float = 0.0001,
                power: float = 1.0,
                cycle: bool = False,
                last_step: int = -1,
                verbose: bool = False,
                )"""

CosineDecayLR_CODE = """oneflow.optim.lr_scheduler.CosineDecayLR(
                optimizer: Optimizer,
                decay_steps: int,
                alpha: float = 0.0,
                last_step: int = -1,
                verbose: bool = False,
                )"""

CosineAnnealingLR_CODE = """oneflow.optim.lr_scheduler.CosineAnnealingLR(
                optimizer: Optimizer,
                T_max: int,
                eta_min: float = 0.0,
                last_step: int = -1,
                verbose: bool = False,
                )"""

CosineAnnealingWarmRestarts_CODE = """oneflow.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer: Optimizer,
                T_0: int,
                T_mult: int = 1,
                eta_min: float = 0.0,
                decay_rate: float = 1.0,
                restart_limit: int = 0,
                last_step: int = -1,
                verbose: bool = False,
                )"""

SequentialLR_CODE = """oneflow.optim.lr_scheduler.SequentialLR(
                optimizer: Optimizer,
                schedulers: Sequence[LRScheduler],
                milestones: Sequence[int],
                interval_rescaling: Union[Sequence[bool], bool] = False,
                last_step: int = -1,
                verbose: bool = False,
                )"""

WarmupLR_CODE = """oneflow.optim.lr_scheduler.WarmupLR(
                scheduler_or_optimizer: Union[LRScheduler, Optimizer],
                warmup_factor: float = 1.0 / 3,
                warmup_iters: int = 5,
                warmup_method: str = "linear",
                warmup_prefix: bool = False,
                last_step=-1,
                verbose=False,
                )"""

ReduceLROnPlateau_CODE = """oneflow.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.1,
                patience=10,
                threshold=1e-4,
                threshold_mode="rel",
                cooldown=0,
                min_lr=0,
                eps=1e-8,
                verbose=False,
                )"""

IS_DISPLAY_CODE = False


def _display(display_steps, steps, lrs):
    # altair
    line = (  # Creating an empty chart in the beginning when the page loads
        alt.Chart(pd.DataFrame({"last_step": [], "lr": []}))
        .mark_line(point={"filled": True, "fill": "red"})
        .encode(
            x=X(
                "last_step",
                axis=Axis(title="step"),
                scale=alt.Scale(domain=[0, steps[-1] + 2]),
            ),
            y=Y(
                "lr",
                axis=Axis(title="lr"),
                scale=alt.Scale(domain=[min(lrs) * 0.8, max(lrs) * 1.2]),
            ),
            color=alt.value("#FFAA00"),
        )
        .properties(width=600, height=400)
        .interactive()
    )
    bar_plot = st.altair_chart(line)

    for i in range(display_steps):
        df = pd.DataFrame({"last_step": steps[: i + 1], "lr": lrs[: i + 1]})
        line = (
            alt.Chart(df)
            .mark_line(point={"filled": True, "fill": "red"})
            .encode(
                x=X(
                    "last_step",
                    axis=Axis(title="step"),
                    scale=alt.Scale(domain=[0, steps[-1] + 2]),
                ),
                y=Y(
                    "lr",
                    axis=Axis(title="lr"),
                    scale=alt.Scale(domain=[min(lrs) * 0.8, max(lrs) * 1.2]),
                ),
                color=alt.value("#FFAA00"),
            )
            .properties(width=600, height=400)
            .interactive()
        )
        bar_plot.altair_chart(line)
        # Pretend we're doing some computation that takes time.
        time.sleep(0.5)


# st.title("Learning Rate Scheduler Visualization")
st.header("Learning Rate Scheduler Visualization")


scheduler = st.selectbox(
    "Please choose one scheduler to display",
    (
        "ConstantLR",
        "LinearLR",
        "ExponentialLR",
        "StepLR",
        "MultiStepLR",
        "PolynomialLR",
        "CosineDecayLR",
        "CosineAnnealingLR",
        "CosineAnnealingWarmRestarts",
        # "LambdaLR",
        # "SequentialLR",
        # "WarmupLR",
        # "ChainedScheduler",
        # "ReduceLROnPlateau",
    ),
)

if scheduler == "ConstantLR":
    if IS_DISPLAY_CODE:
        st.code(ConstantLR_CODE, language="python")
    st.write("You can set argument values")
    factor = st.slider("factor:", 0.0, 1.0, 0.3)
    total_iters = st.slider("total_iters:", 0, 20, 5)
    lr = st.slider("initial learning rate in Optimizer(e.g. SGD, Adam):", 0.0, 1.0, 0.1)

    net = flow.nn.Linear(10, 2)
    optimizer = flow.optim.SGD(net.parameters(), lr=lr)
    scheduler = flow.optim.lr_scheduler.ConstantLR(
        optimizer=optimizer, factor=factor, total_iters=total_iters
    )
    steps = []
    lrs = []
    display_steps = max(6, total_iters * 2)
    for i in range(display_steps):
        steps.append(i)
        lrs.append(scheduler.get_last_lr()[0])
        scheduler.step()

    col1, col2, col3 = st.columns(3)
    if col2.button("Display?"):
        _display(display_steps, steps, lrs)


elif scheduler == "LinearLR":
    if IS_DISPLAY_CODE:
        st.code(LinearLR_CODE, language="python")
    st.write("You can set argument values")
    start_factor = st.slider("start_factor:", 0.0, 1.0, 0.3)
    end_factor = st.slider("end_factor:", 0.0, 1.0, 1.0)
    total_iters = st.slider("total_iters:", 0, 20, 5)
    lr = st.slider("initial learning rate in Optimizer(e.g. SGD, Adam):", 0.0, 1.0, 0.1)

    net = flow.nn.Linear(10, 2)
    optimizer = flow.optim.SGD(net.parameters(), lr=lr)
    scheduler = flow.optim.lr_scheduler.LinearLR(
        optimizer=optimizer,
        start_factor=start_factor,
        end_factor=end_factor,
        total_iters=total_iters,
    )
    steps = []
    lrs = []
    display_steps = max(6, total_iters * 2)
    for i in range(display_steps):
        steps.append(i)
        lrs.append(scheduler.get_last_lr()[0])
        scheduler.step()

    col1, col2, col3 = st.columns(3)
    if col2.button("Display?"):
        _display(display_steps, steps, lrs)

elif scheduler == "ExponentialLR":
    if IS_DISPLAY_CODE:
        st.code(ExponentialLR_CODE, language="python")
    st.write("You can set argument values")
    gamma = st.slider("gamma:", 0.0, 1.0, 0.9)
    lr = st.slider("initial learning rate in Optimizer(e.g. SGD, Adam):", 0.0, 1.0, 0.1)

    net = flow.nn.Linear(10, 2)
    optimizer = flow.optim.SGD(net.parameters(), lr=lr)
    scheduler = flow.optim.lr_scheduler.ExponentialLR(
        optimizer=optimizer,
        gamma=gamma,
    )
    steps = []
    lrs = []
    display_steps = 20
    for i in range(display_steps):
        steps.append(i)
        lrs.append(scheduler.get_last_lr()[0])
        scheduler.step()

    col1, col2, col3 = st.columns(3)
    if col2.button("Display?"):
        _display(display_steps, steps, lrs)

elif scheduler == "StepLR":
    if IS_DISPLAY_CODE:
        st.code(StepLR_CODE, language="python")
    st.write("You can set argument values")
    step_size = st.slider("step_size:", 0, 10, 2)
    gamma = st.slider("gamma:", 0.0, 1.0, 0.9)
    lr = st.slider("initial learning rate in Optimizer(e.g. SGD, Adam):", 0.0, 1.0, 0.1)

    net = flow.nn.Linear(10, 2)
    optimizer = flow.optim.SGD(net.parameters(), lr=lr)
    scheduler = flow.optim.lr_scheduler.StepLR(
        optimizer=optimizer,
        step_size=step_size,
        gamma=gamma,
    )
    steps = []
    lrs = []
    display_steps = 20
    for i in range(display_steps):
        steps.append(i)
        lrs.append(scheduler.get_last_lr()[0])
        scheduler.step()

    col1, col2, col3 = st.columns(3)
    if col2.button("Display?"):
        _display(display_steps, steps, lrs)


elif scheduler == "MultiStepLR":
    if IS_DISPLAY_CODE:
        st.code(MultiStepLR_CODE, language="python")
    st.write("You can set argument values")

    collect_numbers = lambda x: [int(i) for i in re.split("[^0-9]", x) if i != ""]
    milestones = st.text_input("PLease enter milestones")
    milestones = collect_numbers(milestones)
    if milestones is None or len(milestones) == 0:
        milestones = [5]
    gamma = st.slider("gamma:", 0.0, 1.0, 0.9)
    lr = st.slider("initial learning rate in Optimizer(e.g. SGD, Adam):", 0.0, 1.0, 0.1)

    net = flow.nn.Linear(10, 2)
    optimizer = flow.optim.SGD(net.parameters(), lr=lr)
    scheduler = flow.optim.lr_scheduler.MultiStepLR(
        optimizer=optimizer,
        milestones=milestones,
        gamma=gamma,
    )
    steps = []
    lrs = []
    display_steps = milestones[-1] + 5
    for i in range(display_steps):
        steps.append(i)
        lrs.append(scheduler.get_last_lr()[0])
        scheduler.step()

    col1, col2, col3 = st.columns(3)
    if col2.button("Display?"):
        _display(display_steps, steps, lrs)

elif scheduler == "PolynomialLR":
    if IS_DISPLAY_CODE:
        st.code(PolynomialLR_CODE, language="python")
    st.write("You can set argument values")
    steps = st.slider("steps:", 1, 10, 5)
    end_learning_rate = st.slider("end_learning_rate", 0.0, 1.0, 0.0001)
    power = st.slider("power", 0.0, 10.0, 1.0)
    cycle = st.checkbox(
        "cycle",
    )
    lr = st.slider("initial learning rate in Optimizer(e.g. SGD, Adam):", 0.0, 1.0, 0.1)

    net = flow.nn.Linear(10, 2)
    optimizer = flow.optim.SGD(net.parameters(), lr=lr)
    scheduler = flow.optim.lr_scheduler.PolynomialLR(
        optimizer=optimizer,
        steps=steps,
        end_learning_rate=end_learning_rate,
        power=power,
        cycle=cycle,
    )
    x_steps = []
    lrs = []
    display_steps = max(steps + 5, 10)
    for i in range(display_steps):
        x_steps.append(i)
        lrs.append(scheduler.get_last_lr()[0])
        scheduler.step()

    col1, col2, col3 = st.columns(3)
    if col2.button("Display?"):
        _display(display_steps, x_steps, lrs)

elif scheduler == "CosineDecayLR":
    if IS_DISPLAY_CODE:
        st.code(CosineDecayLR_CODE, language="python")
    st.write("You can set argument values")
    decay_steps = st.slider("decay_steps:", 0, 10, 5)
    alpha = st.slider("alpha", 0.0, 1.0, 0.0)
    lr = st.slider("initial learning rate in Optimizer(e.g. SGD, Adam):", 0.0, 1.0, 0.1)

    net = flow.nn.Linear(10, 2)
    optimizer = flow.optim.SGD(net.parameters(), lr=lr)
    scheduler = flow.optim.lr_scheduler.CosineDecayLR(
        optimizer=optimizer,
        decay_steps=decay_steps,
        alpha=alpha,
    )
    x_steps = []
    lrs = []
    display_steps = max(decay_steps + 5, 10)
    for i in range(display_steps):
        x_steps.append(i)
        lrs.append(scheduler.get_last_lr()[0])
        scheduler.step()

    col1, col2, col3 = st.columns(3)
    if col2.button("Display?"):
        _display(display_steps, x_steps, lrs)

elif scheduler == "CosineAnnealingLR":
    if IS_DISPLAY_CODE:
        st.code(CosineAnnealingLR_CODE, language="python")
    st.write("You can set argument values")
    T_max = st.slider("T_max", 1, 20, 20)
    eta_min = st.slider("eta_min", 0.0, 1.0, 0.0)
    lr = st.slider("initial learning rate in Optimizer(e.g. SGD, Adam):", 0.0, 1.0, 0.1)

    net = flow.nn.Linear(10, 2)
    optimizer = flow.optim.SGD(net.parameters(), lr=lr)
    scheduler = flow.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=T_max,
        eta_min=eta_min,
    )
    x_steps = []
    lrs = []
    display_steps = max(T_max + 5, 20)
    for i in range(display_steps):
        x_steps.append(i)
        lrs.append(scheduler.get_last_lr()[0])
        scheduler.step()

    col1, col2, col3 = st.columns(3)
    if col2.button("Display?"):
        _display(display_steps, x_steps, lrs)

elif scheduler == "CosineAnnealingWarmRestarts":
    if IS_DISPLAY_CODE:
        st.code(CosineAnnealingWarmRestarts_CODE, language="python")
    st.write("You can set argument values")
    T_0 = st.slider("T_0", 1, 20, 5)
    T_mult = st.slider("T_mult", 1, 5, 1)
    eta_min = st.slider("eta_min", 0.0, 1.0, 0.0)
    decay_rate = st.slider("decay_rate", 0.0, 1.0, 1.0)
    restart_limit = st.slider("restart_limit", 0, 5, 0)
    lr = st.slider("initial learning rate in Optimizer(e.g. SGD, Adam):", 0.0, 1.0, 0.1)

    net = flow.nn.Linear(10, 2)
    optimizer = flow.optim.SGD(net.parameters(), lr=lr)
    scheduler = flow.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer=optimizer,
        T_0=T_0,
        T_mult=T_mult,
        eta_min=eta_min,
        decay_rate=decay_rate,
        restart_limit=restart_limit,
    )
    x_steps = []
    lrs = []
    display_steps = max(T_0 + 5, 20)
    for i in range(display_steps):
        x_steps.append(i)
        lrs.append(scheduler.get_last_lr()[0])
        scheduler.step()

    col1, col2, col3 = st.columns(3)
    if col2.button("Display?"):
        _display(display_steps, x_steps, lrs)

# elif scheduler == "LambdaLR":
#     code = """oneflow.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_step=-1, verbose=False)"""
#     st.code(code, language="python")

elif scheduler == "SequentialLR":
    if IS_DISPLAY_CODE:
        st.code(SequentialLR_CODE, language="python")
    st.write("You can set argument values")
    schedulers = st.multiselect(
        "you can choose multiple schedulers",
        [
            "ConstantLR",
            "LinearLR",
            "ExponentialLR",
            "StepLR",
            "MultiStepLR",
            "PolynomialLR",
            "CosineDecayLR",
            "CosineAnnealingLR",
            "CosineAnnealingWarmRestarts",
            "ConstantLR",
            "LinearLR",
            "ExponentialLR",
            "StepLR",
            "MultiStepLR",
            "PolynomialLR",
            "CosineDecayLR",
            "CosineAnnealingLR",
            "CosineAnnealingWarmRestarts",
        ],
    )
    collect_numbers = lambda x: [int(i) for i in re.split("[^0-9]", x) if i != ""]
    milestones = st.text_input("PLease enter milestones")
    milestones = collect_numbers(milestones)
    interval_rescaling = st.checkbox("interval_rescaling")
    lr = st.slider("initial learning rate in Optimizer(e.g. SGD, Adam):", 0.0, 1.0, 0.1)

    net = flow.nn.Linear(10, 2)
    optimizer = flow.optim.SGD(net.parameters(), lr=lr)
    scheduler = flow.optim.lr_scheduler.SequentialLR(
        optimizer=optimizer,
        schedulers=schedulers,
        milestones=milestones,
        interval_rescaling=interval_rescaling,
    )
    x_steps = []
    lrs = []
    display_steps = max(milestones[-1] + 5, 20)
    for i in range(display_steps):
        x_steps.append(i)
        lrs.append(scheduler.get_last_lr()[0])
        scheduler.step()

    col1, col2, col3 = st.columns(3)
    if col2.button("Display?"):
        _display(display_steps, x_steps, lrs)

elif scheduler == "WarmupLR":
    if IS_DISPLAY_CODE:
        st.code(WarmupLR_CODE, language="python")
    scheduler_or_optimizer = st.selectbox(
        "choose one scheduler for scheduler_or_optimizer",
        [
            "ConstantLR",
            "LinearLR",
            "ExponentialLR",
            "StepLR",
            "MultiStepLR",
            "PolynomialLR",
            "CosineDecayLR",
            "CosineAnnealingLR",
            "CosineAnnealingWarmRestarts",
        ],
    )
    warmup_factor = st.slider("warmup_factor:", 0.0, 1.0, 0.3)
    warmup_iters = st.slider("warmup_iters:", 1, 10, 5)
    warmup_method = st.selectbox("warmup_method", ["linear", "constant"])
    warmup_prefix = st.checkbox("warmup_prefix")
    lr = st.slider("initial learning rate in Optimizer(e.g. SGD, Adam):", 0.0, 1.0, 0.1)

    net = flow.nn.Linear(10, 2)
    optimizer = flow.optim.SGD(net.parameters(), lr=lr)
    scheduler = flow.optim.lr_scheduler.WarmupLR(
        optimizer=optimizer,
        scheduler_or_optimizer=scheduler_or_optimizer,
        warmup_factor=warmup_factor,
        warmup_iters=warmup_iters,
        warmup_method=warmup_method,
        warmup_prefix=warmup_prefix,
    )
    x_steps = []
    lrs = []
    display_steps = max(warmup_factor + 5, 20)
    for i in range(display_steps):
        x_steps.append(i)
        lrs.append(scheduler.get_last_lr()[0])
        scheduler.step()

    col1, col2, col3 = st.columns(3)
    if col2.button("Display?"):
        _display(display_steps, x_steps, lrs)


elif scheduler == "ChainedScheduler":
    if IS_DISPLAY_CODE:
        code = """oneflow.optim.lr_scheduler.ChainedScheduler(schedulers)"""
    st.code(code, language="python")
    st.write("You can set argument values")
    schedulers = st.multiselect(
        "you can choose multiple schedulers",
        [
            "ConstantLR",
            "LinearLR",
            "ExponentialLR",
            "StepLR",
            "MultiStepLR",
            "PolynomialLR",
            "CosineDecayLR",
            "CosineAnnealingLR",
            "CosineAnnealingWarmRestarts",
            "ConstantLR",
            "LinearLR",
            "ExponentialLR",
            "StepLR",
            "MultiStepLR",
            "PolynomialLR",
            "CosineDecayLR",
            "CosineAnnealingLR",
            "CosineAnnealingWarmRestarts",
        ],
    )
    lr = st.slider("initial learning rate in Optimizer(e.g. SGD, Adam):", 0.0, 1.0, 0.1)

    net = flow.nn.Linear(10, 2)
    optimizer = flow.optim.SGD(net.parameters(), lr=lr)
    scheduler = flow.optim.lr_scheduler.ChainedScheduler(
        optimizer=optimizer,
        schedulers=schedulers,
    )
    x_steps = []
    lrs = []
    display_steps = 20
    for i in range(display_steps):
        x_steps.append(i)
        lrs.append(scheduler.get_last_lr()[0])
        scheduler.step()

    col1, col2, col3 = st.columns(3)
    if col2.button("Display?"):
        _display(display_steps, x_steps, lrs)

# elif scheduler == "ReduceLROnPlateau":
#     st.code(ReduceLROnPlateau_CODE, language="python")
#     st.write("You can set argument values")
#     mode = st.selectbox(
#         "mode",
#         [
#             "min",
#             "max",
#         ],
#     )
#     factor = st.slider("factor", 1e-5, 1.0 - 1e-5, 0.1)
#     patience = st.slider("patience", 1, 20, 10)
#     threshold = st.slider("threshold", 1e-4, 9e-4, 1e-4)
#     threshold_mode = st.selectbox("threshold_mode", ["rel", "abs"])
#     cooldown = st.slider("cooldown", 0, 10, 0)
#     min_lr = st.slider("min_lr", 0.0, 1.0, 0.0)
#     eps = st.slider("eps", 1e-8, 9e-8, 1e-8)
#     lr = st.slider("initial learning rate in Optimizer(e.g. SGD, Adam):", 0.0, 1.0, 0.1)

#     net = flow.nn.Linear(10, 2)
#     optimizer = flow.optim.SGD(net.parameters(), lr=lr)
#     scheduler = flow.optim.lr_scheduler.ReduceLROnPlateau(
#         optimizer=optimizer,
#         mode=mode,
#         factor=factor,
#         patience=patience,
#         threshold=threshold,
#         threshold_mode=threshold_mode,
#         cooldown=cooldown,
#         min_lr=min_lr,
#         eps=eps,
#     )
#     x_steps = []
#     lrs = []
#     display_steps = 25
#     for i in range(display_steps):
#         x_steps.append(i)
#         lrs.append(scheduler.get_last_lr()[0])
#         scheduler.step()

#     col1, col2, col3 = st.columns(3)
#     if col2.button("Display?"):
#         _display(display_steps, x_steps, lrs)
