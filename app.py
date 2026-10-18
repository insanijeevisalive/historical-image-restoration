
import streamlit as st
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

from restoration import (
    ORIGINAL_DIR,
    DAMAGED_DIR,
    MASK_DIR,
    RESTORED_DIR,
    METRICS_DIR,
    IMAGE_EXTENSIONS,
    read_image,
    resize_image,
    create_damage,
    restore_image,
    calculate_metrics,
)

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Historical Image Restoration",
    page_icon="🖼️",
    layout="wide"
)

st.title("Historical Image Restoration")
st.caption(
    "Historical Image Restoration Using Digital Image Processing Techniques"
)

st.info(
    "Select a numbered dataset image or upload your own photograph. "
    "Both options use the same synthetic damage and restoration pipeline."
)


# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def to_rgb(image):
    """Convert OpenCV BGR image to RGB for display."""
    if image is None:
        raise ValueError("Cannot display an empty image.")

    if image.ndim == 2:
        return image

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def save_image(path, image):
    """Save an image and handle errors."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if image is None:
        raise ValueError(f"Image is empty: {path}")

    if not cv2.imwrite(str(path), image):
        raise IOError(f"Could not save image: {path}")


def process_image(
    original,
    output_name,
    display_name,
    seed,
    max_size,
    save_outputs
):
    """Generate damage, restore the image, calculate metrics and save outputs."""

    original = resize_image(
        original,
        max_size=max_size
    )

    # 1. Generate synthetic damage
    damaged, mask = create_damage(
        original,
        seed=int(seed)
    )

    # 2. Restore synthetic damage
    restored, final_image = restore_image(
        damaged,
        mask
    )

    if restored is None or final_image is None:
        raise ValueError("The restoration algorithm returned an empty image.")

    # 3. Calculate metrics against the clean reference
    metrics = pd.DataFrame([
        {
            "Stage": "Damaged",
            **calculate_metrics(original, damaged)
        },
        {
            "Stage": "Restored",
            **calculate_metrics(original, restored)
        },
        {
            "Stage": "Final output",
            **calculate_metrics(original, final_image)
        }
    ])

    # 4. Save outputs
    if save_outputs:
        stem = Path(output_name).stem

        save_image(
            DAMAGED_DIR / f"{stem}_damaged.png",
            damaged
        )

        save_image(
            MASK_DIR / f"{stem}_mask.png",
            mask
        )

        save_image(
            RESTORED_DIR / f"{stem}_restored.png",
            restored
        )

        save_image(
            RESTORED_DIR / f"{stem}_final.png",
            final_image
        )

        metrics.to_csv(
            METRICS_DIR / f"{stem}_metrics.csv",
            index=False
        )

    return {
        "name": display_name,
        "output_name": output_name,
        "original": original,
        "damaged": damaged,
        "mask": mask,
        "restored": restored,
        "final": final_image,
        "metrics": metrics
    }


# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

files = sorted(
    [
        path
        for path in ORIGINAL_DIR.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    ],
    key=lambda path: path.name.lower()
)

# Create stable numbered labels for the current sorted dataset.
image_labels = {
    path: f"Image {index}"
    for index, path in enumerate(files)
}


# --------------------------------------------------
# SIDEBAR SETTINGS
# --------------------------------------------------

with st.sidebar:
    st.header("Dataset settings")

    st.write(f"Images found: **{len(files)}**")

    seed = st.number_input(
        "Damage seed",
        min_value=0,
        max_value=999999,
        value=42,
        step=1,
        help="Use the same seed to reproduce the same synthetic damage."
    )

    max_size = st.selectbox(
        "Maximum image dimension",
        [256, 384, 512, 640, 768],
        index=2,
        help="Limits the longest image dimension. Smaller images are not enlarged."
    )

    save_outputs = st.checkbox(
        "Save generated results",
        value=True
    )


# --------------------------------------------------
# IMAGE SOURCE SELECTION
# --------------------------------------------------

st.subheader("Choose your image source")

mode = st.radio(
    "Image source",
    [
        "Dataset Image",
        "Upload Your Own Image"
    ],
    horizontal=True
)


# --------------------------------------------------
# OPTION 1: DATASET IMAGE
# --------------------------------------------------

if mode == "Dataset Image":

    if not files:
        st.warning(f"No dataset images found in: {ORIGINAL_DIR}")
        st.info("Place your clean images inside dataset/original/.")
        st.stop()

    selected = st.selectbox(
        "Select dataset image",
        files,
        format_func=lambda path: image_labels[path]
    )

    selected_label = image_labels[selected]

    st.caption(f"Selected: **{selected_label}**")

    if st.button(
        "Restore and evaluate",
        type="primary",
        use_container_width=True,
        key="dataset_run"
    ):
        try:
            with st.spinner(f"Processing {selected_label}..."):

                original = read_image(selected)

                result = process_image(
                    original=original,
                    output_name=f"{selected_label.replace(' ', '_')}.png",
                    display_name=selected_label,
                    seed=seed,
                    max_size=max_size,
                    save_outputs=save_outputs
                )

                st.session_state["restoration_result"] = result
                st.session_state["result_source"] = mode

            st.success(f"{selected_label} processed successfully.")

        except Exception as exc:
            st.error(f"Could not process {selected_label}: {exc}")


# --------------------------------------------------
# OPTION 2: UPLOAD YOUR OWN IMAGE
# --------------------------------------------------

elif mode == "Upload Your Own Image":

    st.write("Upload a photograph to test the restoration pipeline.")

    st.warning(
        "This version creates synthetic damage on your uploaded image "
        "and restores that generated damage. It does not yet automatically "
        "remove scratches or stains that were already present."
    )

    uploaded_file = st.file_uploader(
        "Upload a historical photograph",
        type=["jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp"]
    )

    if uploaded_file is not None:

        try:
            file_bytes = np.frombuffer(
                uploaded_file.getvalue(),
                dtype=np.uint8
            )

            uploaded_image = cv2.imdecode(
                file_bytes,
                cv2.IMREAD_COLOR
            )

            if uploaded_image is None:
                st.error("Could not read the uploaded image.")

            else:
                st.markdown("**Uploaded original image**")

                preview = resize_image(
                    uploaded_image,
                    max_size=max_size
                )

                st.image(
                    to_rgb(preview),
                    use_container_width=True
                )

                upload_name = Path(uploaded_file.name).name

                if st.button(
                    "Restore and evaluate uploaded image",
                    type="primary",
                    use_container_width=True,
                    key="upload_run"
                ):
                    with st.spinner("Processing uploaded image..."):

                        result = process_image(
                            original=uploaded_image,
                            output_name=f"upload_{upload_name}",
                            display_name=upload_name,
                            seed=seed,
                            max_size=max_size,
                            save_outputs=save_outputs
                        )

                        st.session_state["restoration_result"] = result
                        st.session_state["result_source"] = mode

                    st.success("Uploaded image processed successfully.")

        except Exception as exc:
            st.error(f"Could not process uploaded image: {exc}")


# --------------------------------------------------
# DISPLAY RESULTS
# --------------------------------------------------

result = st.session_state.get("restoration_result")
result_source = st.session_state.get("result_source")

if result is not None and result_source == mode:

    st.divider()

    st.subheader(f"Results: {result['name']}")

    views = [
        ("Clean original", result["original"]),
        ("Synthetic damaged image", result["damaged"]),
        ("Ground-truth damage mask", result["mask"]),
        ("Restored image", result["restored"]),
        ("Final output", result["final"])
    ]

    columns = st.columns(2)

    for index, (label, image) in enumerate(views):

        with columns[index % 2]:

            st.markdown(f"**{label}**")

            if image.ndim == 2:
                st.image(
                    image,
                    clamp=True,
                    use_container_width=True
                )
            else:
                st.image(
                    to_rgb(image),
                    use_container_width=True
                )

    # ----------------------------------------------
    # QUALITY METRICS
    # ----------------------------------------------

    st.divider()
    st.subheader("Image quality metrics")

    st.caption(
        "Metrics compare generated outputs against the clean reference. "
        "Lower MSE and higher PSNR/SSIM generally indicate greater similarity."
    )

    st.dataframe(
        result["metrics"],
        use_container_width=True,
        hide_index=True
    )

    # ----------------------------------------------
    # DOWNLOADS
    # ----------------------------------------------

    st.subheader("Download results")

    download_col1, download_col2 = st.columns(2)

    with download_col1:

        success, encoded = cv2.imencode(
            ".png",
            result["final"]
        )

        if success:
            st.download_button(
                "Download restored image",
                data=encoded.tobytes(),
                file_name=f"{Path(result['output_name']).stem}_restored.png",
                mime="image/png",
                use_container_width=True
            )

    with download_col2:

        csv_data = result["metrics"].to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "Download metrics CSV",
            data=csv_data,
            file_name=f"{Path(result['output_name']).stem}_metrics.csv",
            mime="text/csv",
            use_container_width=True
        )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Historical Image Restoration | Dataset experiments and image quality evaluation"
)
