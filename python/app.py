from pathlib import Path
import torch
import streamlit as st
import numpy as np
from torchvision.utils import make_grid
from sklearn.metrics import f1_score

from data import CifarDataset, build_loader


########################################
#           Side bar Section           #
########################################
st.sidebar.title("Corruption Scenario")

# corrupt_domain = st.sidebar.selectbox(
#     "Corrupt Domain",
#     options=[
#         "all",
#         "brightness",
#         "contrast",
#         "defocus_blur",
#         "elastic_transform",
#         "fog",
#         "frost",
#         "gaussian_noise",
#         "glass_blur",
#         "impulse_noise",
#         "jpeg_compression",
#         "motion_blur",
#         "pixelate",
#         "shot_noise",
#         "snow",
#         "test",
#         "zoom_blur",
#     ],
# )
severity = st.sidebar.slider("Severity Level", min_value=1, max_value=5, value=5)
num_samples = st.sidebar.number_input("samples per domain", value=100)
batch_size = st.sidebar.number_input("batch size", value=100)
model_checkpoint = st.sidebar.selectbox(
    "model_checkpoint", options=["pre-trained", "src_0", "src_1", "src_2"]
)

########################################
#          Main Page Section           #
########################################
# Session State
total_samples = num_samples * 15
num_steps = total_samples // batch_size
num_steps += 1 if total_samples % batch_size else 0
default_values = {
    "true_cls_list": [],
    "pred_cls_list": [],
    "accuracy_list": [],
    "distance_l2_list": [],
    "current_step": 0,
    "num_steps": num_steps,
}

for key, value in default_values.items():
    st.session_state.setdefault(key, value)

st.title("TTA Demo App")


# Data & Model Setting
## Load Dataset
@st.cache_resource
def load_dataloader(severity: int, num_samples: int, batch_size: int):
    dataset = CifarDataset(
        severity=severity,
        num_samples=num_samples,
    )
    loader = build_loader(dataset, batch_size=batch_size)
    return iter(loader)


dataloader = load_dataloader(
    severity=severity,
    num_samples=num_samples,
    batch_size=batch_size,
)


## Load Model
@st.cache_resource
def load_model():
    import torchvision
    import torch.nn as nn

    if model_checkpoint == "pre-trained":
        net = torchvision.models.resnet18(pretrained=True)
        num_feats = net.fc.in_features
        net.fc = nn.Linear(num_feats, 10)  # match class number
    else:
        checkpoint_path = Path(__file__).parent.parent / "tgt_test"
        checkpoint_path / f"reproduce_{model_checkpoint}" / "cp/cp_last.pth.tar"

    return net.to(device)


def load_next_batch():
    feats, cls, dos = next(dataloader)
    return feats, cls, dos


device = "mps"
net = load_model()


## Calculate accuracy
@torch.no_grad()
def get_acc(current_batch):
    feats, cls, dls = current_batch
    feats, cls = feats.to(device), cls.to(device)
    y_pred = net(feats)
    y_pred = y_pred.max(1, keepdim=False)[1]

    # append values to lists
    st.session_state.true_cls_list += [int(c) for c in cls]
    st.session_state.pred_cls_list += [int(c) for c in y_pred.tolist()]
    match = sum(
        np.array(st.session_state.true_cls_list)
        == np.array(st.session_state.pred_cls_list)
    )
    cumul_accuracy = match / len(st.session_state.true_cls_list)
    st.session_state.accuracy_list.append(cumul_accuracy)
    return cumul_accuracy


## Scenario
st.subheader("Step-by-Step Time Test Adaption")

### Progress bar
st.subheader(f"Step: {st.session_state.current_step + 1}")
bar = st.progress(st.session_state.current_step / num_steps)

### Next button
button_disabled = st.session_state.current_step >= num_steps - 1
next_button = st.button("Next Step", disabled=button_disabled)
if st.button("Reset"):
    for key, value in default_values.items():
        st.session_state[key] = value

if next_button:
    # get next batch
    current_batch = load_next_batch()
    feats, *_ = current_batch
    st.image(
        make_grid(feats, nrow=10).permute(1, 2, 0).numpy(), use_container_width=True
    )
    # get accuracy
    cumsum_acc = get_acc(current_batch)
    # update step
    step = min(st.session_state.current_step + 1, st.session_state.num_steps)
    st.session_state.current_step = step
    # update metrics
    st.line_chart(st.session_state.accuracy_list, x_label="step", y_label="cum_acc")
    st.metric("Cumulation Accuracy", cumsum_acc)
    bar.progress(st.session_state.current_step / num_steps)
