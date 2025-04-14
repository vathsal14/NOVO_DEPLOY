import streamlit as st
import numpy as np
import pickle
import nibabel as nib
import tempfile
import os
import shutil
from nilearn import image, datasets, masking
import dicom2nifti
import pydicom

# Load the trained model and scalers
model_path = "model/Parkinson_Model.pkl"
scaler_X_path = "model/scaler.pkl"
scaler_y_path = "model/scaler_y.pkl"

# Load model and scalers
final_model = pickle.load(open(model_path, 'rb'))
scaler_X = pickle.load(open(scaler_X_path, 'rb'))
scaler_y = pickle.load(open(scaler_y_path, 'rb'))

# Streamlit App Title
st.set_page_config(page_title="🧠 Parkinson's Risk Prediction from DATSCAN", layout="centered")
st.title("🧠 Parkinson's Risk Prediction App from DATSCAN Scan")
st.markdown("Upload a `.dcm` or `.nii.gz` DATSCAN scan to predict Parkinson's risk.")

# Upload DATSCAN file
uploaded_file = st.file_uploader("📤 Upload DATSCAN File (.dcm or .nii.gz)", type=["dcm", "nii", "nii.gz"])

# Manual feature inputs
NP3TOT = st.number_input("NP3TOT (Motor Symptoms Score)", min_value=0, format="%d")
UPSIT_PRCNTGE = st.number_input("UPSIT_PRCNTGE (Smell Test Score)", min_value=0.0, format="%.2f")
COGCHG = st.selectbox("Cognitive Change (0 = No, 1 = Yes)", [0, 1])

if uploaded_file is not None:
    try:
        # Save uploaded file to a temporary directory
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = os.path.join(tmp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.read())

            # Check if DICOM
            is_dicom = False
            try:
                dcm = pydicom.dcmread(file_path)
                is_dicom = True
            except:
                pass

            # Convert DICOM to NIfTI if needed
            if is_dicom:
                dicom_folder = os.path.join(tmp_dir, "dicom")
                os.makedirs(dicom_folder, exist_ok=True)
                shutil.move(file_path, os.path.join(dicom_folder, "image.dcm"))

                nifti_output = os.path.join(tmp_dir, "converted.nii.gz")
                dicom2nifti.convert_directory(dicom_folder, tmp_dir)
                datscan_img = nib.load(nifti_output)
            else:
                datscan_img = nib.load(file_path)

            datscan_data = datscan_img.get_fdata()

            # Compute brain mask
            brain_mask = masking.compute_brain_mask(datscan_img)
            brain_data = datscan_data[brain_mask.get_fdata() > 0]

            # Load Harvard-Oxford Subcortical Atlas
            st.info("Loading Harvard-Oxford Subcortical Atlas...")
            atlas = datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr25-1mm')
            atlas_img = atlas.filename
            atlas_labels = atlas.labels

            # Resample atlas to match DATSCAN image
            atlas_resampled = image.resample_to_img(atlas_img, datscan_img, interpolation='nearest')
            atlas_data = atlas_resampled.get_fdata()

            # Extract ROI values
            roi_values = {}

            for i, label in enumerate(atlas_labels):
                if not label:
                    continue

                region_mask = (atlas_data == i)
                region_voxels = datscan_data[region_mask]

                if region_voxels.size == 0:
                    continue

                region_mean = np.mean(region_voxels)

                # Normalize region mean value by the brain mean
                normalized_value = region_mean / np.mean(brain_data)

                if "Putamen" in label or "Caudate" in label:
                    roi_values[label] = normalized_value

            # Extracted normalized Putamen and Caudate values
            putamen_r = roi_values.get('Right Putamen', 0.0)
            putamen_l = roi_values.get('Left Putamen', 0.0)
            caudate_r = roi_values.get('Right Caudate', 0.0)
            caudate_l = roi_values.get('Left Caudate', 0.0)

            # Display extracted values
            st.success("✅ Brain region values extracted.")
            st.markdown("### DATSCAN values")
            st.write(f"**Right Putamen:** {putamen_r:.2f}")
            st.write(f"**Left Putamen:** {putamen_l:.2f}")
            st.write(f"**Right Caudate:** {caudate_r:.2f}")
            st.write(f"**Left Caudate:** {caudate_l:.2f}")

            # Combine the extracted values with manual inputs
            st.write("""
            ### Using these extracted and manually entered values for Parkinson's Risk Prediction:
            """)

            # Input the extracted values into the prediction model
            input_data = np.array([[putamen_r, putamen_l, caudate_r, caudate_l, NP3TOT, UPSIT_PRCNTGE, COGCHG]])

            # Apply feature scaling
            input_data_scaled = scaler_X.transform(input_data)

            # Make Prediction
            pred_scaled = final_model.predict(input_data_scaled)

            # Convert prediction back to original scale
            risk_percent = float(scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))[0])

            # Risk Status Classification
            risk_status = "No significant risk detected (Negative)" if risk_percent <= 20 else "Significant risk detected (Positive)"

            # Display Results
            st.success(f"🧠 **Predicted Parkinson's Risk Percentage:** {risk_percent:.2f}%")
            st.info(f"**Risk Status:** {risk_status}")

    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
