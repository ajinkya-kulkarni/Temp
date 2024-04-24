import os
import shutil
import sys
import time
from datetime import datetime

import streamlit as st
from PIL import Image
from io import BytesIO
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from skimage.measure import regionprops
from skimage.segmentation import relabel_sequential

# Clear terminal screen
os.system('clear')

# Don't generate the __pycache__ folder locally
sys.dont_write_bytecode = True 
# Print exception without the built-in python warning
sys.tracebacklimit = 0

from modules import *  # Import your custom modules

# Define a random label color map
lbl_cmap = random_label_cmap()

###########################################################################################

model_path = 'OrganoidsTrainedModel'

window_size = 800
overlap = int(0.5 * window_size)

###########################################################################################

from cellpose import models

def predict_mask_from_image_cellpose(normalized_img_patch, gpu_usage = False):

	model = models.CellposeModel(gpu = gpu_usage, pretrained_model = model_path)
	
	channels = [[0, 0]]

	mask, flow, style = model.eval(normalized_img_patch, diameter=None, channels=channels)

	# Check if the mask is empty (all zeros)
	if np.any(mask):
		# Return the predicted mask if it's not empty
		return mask.astype('uint16')
	else:
		# Return an array of zeros if the mask is empty
		return np.zeros(normalized_img_patch.shape, dtype=np.uint16)

###########################################################################################

def analyze_mask(mask, original_img, filename):

	# Ensure mask is uint16
	mask = mask.astype(np.uint16)

	# Calculate properties of each labeled region
	props = regionprops(mask, intensity_image=original_img)

	# Number of labels
	num_labels = len(props)

	# Total area of the labels
	total_area = sum([prop.area for prop in props])

	# Mean area of the labels
	mean_area = np.mean([prop.area for prop in props]) if num_labels > 0 else 0
	mean_area = round(mean_area, 2)

	# Live/dead classification
	live_labels = [prop.label for prop in props if prop.mean_intensity > live_dead_threshold]
	dead_labels = [prop.label for prop in props if prop.mean_intensity <= live_dead_threshold]

	no_live_organoids = len(live_labels)
	no_dead_organoids = len(dead_labels)

	mean_area_live_organoids = total_area / no_live_organoids if no_live_organoids > 0 else 0
	mean_area_live_organoids = round(mean_area_live_organoids, 2)

	mean_area_dead_organoids = total_area / no_dead_organoids if no_dead_organoids > 0 else 0
	mean_area_dead_organoids = round(mean_area_dead_organoids, 2)

	result = [filename, num_labels, total_area, mean_area, no_live_organoids, no_dead_organoids, mean_area_live_organoids, mean_area_dead_organoids]

	return result

###########################################################################################

with open("logo.jpg", "rb") as f:
	image_data = f.read()

image_bytes = BytesIO(image_data)

st.set_page_config(page_title = 'OrganoIDNet', page_icon = image_bytes, layout = "wide", initial_sidebar_state = "expanded", menu_items = {'About': 'This is a application for demonstrating the OrganoIDNet package. Developed, tested and maintained by Ajinkya Kulkarni: https://github.com/ajinkya-kulkarni at the MPI-NAT, Goettingen. Email: kulkajinkya@gmail.com'})

# Title of the web app

st.title(':blue[Analyze Organoids using OrganoIDNet]')

with st.form(key = 'form1', clear_on_submit = False):

	# Get current timestamp
	# timestamp = datetime.now().strftime("%Y%m%d_%H%M")
	timestamp = datetime.now().strftime("%Y%m%d_%H%M")

	# Concatenate timestamp with 'Results_'
	output_dir = f'Results_{timestamp}'

	live_dead_threshold_default = 50

	# Create a slider input field in Streamlit
	live_dead_threshold = st.number_input(
		"Live/Dead Threshold for Organoid health (0 to 255)",
		min_value=0,  # Minimum allowed value
		max_value=255,  # Maximum allowed value
		value=live_dead_threshold_default,  # Default value
		step=1,  # Step size for increments and decrements,
		format='%d')

	live_dead_threshold = int(live_dead_threshold)

	# Multiple file uploader allows user to add their own TIF images
	uploaded_files = st.file_uploader("Drag and drop folder containing the images", type=["tif", "tiff"], accept_multiple_files=True)

	submitted = st.form_submit_button('Analyze')

	####################################################################################

	if uploaded_files is None:
		st.stop()

	####################################################################################

	if submitted:

		total_files = len(uploaded_files)
		if total_files == 0:
			st.stop()

		if os.path.exists(output_dir):
			shutil.rmtree(output_dir)
		os.makedirs(output_dir, exist_ok=True)

		####################################################################################

		progress_bar = st.progress(0)  # Initialize the progress bar at 0

		status_message1 = st.empty()  # Placeholder for displaying messages
		status_message2 = st.empty()  # Placeholder for displaying messages

		image_rendering = st.empty()

		for index, uploaded_file in enumerate(uploaded_files):

			progress_bar.progress((index + 1) / total_files)

			status_message1.write(f'Analyzing {uploaded_file.name}')
			status_message2.write(f'Processing {index + 1}/{total_files} images')

			###########################################################################################

			# Read the uploaded file into a PIL Image object and convert to grayscale

			image = Image.open(uploaded_file).convert('L')
			
			# Convert the image to a numpy array
			image_array = np.array(image)

			normalized_img = read_image_as_grayscale_then_MinMax_normalize(image_array)

			###########################################################################################

			patches, window_coords = patchify(normalized_img, window_size, overlap)
			
			###########################################################################################
			
			predicted_labels = []

			for patch in patches:
			
				label = predict_mask_from_image_cellpose(patch, gpu_usage = True)
			
				smoothed_label = smooth_segmented_labels(label)
			
				predicted_labels.append(smoothed_label)

			###########################################################################################
			
			border_cleaned_predicted_labels = []
			
			for patch, patch_coords in zip(predicted_labels, window_coords):
				cleaned_patch = remove_border_labels(patch, patch_coords, normalized_img)
				border_cleaned_predicted_labels.append(cleaned_patch)
				
			###########################################################################################
			
			region_info_list = compile_label_info(np.array(border_cleaned_predicted_labels), window_coords)
			
			###########################################################################################
			
			# First, extract the bounding boxes from each region in the region_info_list
			# This creates an array of bounding boxes where each box is defined by [x_min, y_min, x_max, y_max]
			boxes = np.array([region['global_bbox'] for region in region_info_list])
			
			# Apply the Non-Maximum Suppression (NMS) function to these boxes.
			# NMS will analyze these bounding boxes and return the indices of boxes that should be kept
			# based on the overlap threshold of 0.5. Boxes that overlap more than this threshold with a larger box
			# will be filtered out.
			nms_indices = non_maximum_suppression(boxes, overlapThresh=0.5)
			
			# Using the indices obtained from NMS, construct the final list of regions.
			# This list will only include regions whose bounding boxes were selected by the NMS process,
			# effectively filtering out regions with significantly overlapping bounding boxes.
			nms_region_info_list = [region_info_list[i] for i in nms_indices]
			
			# final_region_info_list now contains the refined list of regions after applying NMS.
			# These are the regions that are considered significant based on their size and the lack of substantial
			# overlap with larger regions.
			
			###########################################################################################
			
			canvas_old = place_labels_on_canvas(normalized_img, nms_region_info_list)
			
			canvas, _, _ = relabel_sequential(canvas_old)

			###########################################################################################

			mask_analysis = analyze_mask(canvas, image_array, uploaded_file.name)
			
			###########################################################################################
			
			# Create a figure with 4 subplots
			fig, axes = plt.subplots(1, 4, figsize=(20, 5))  # Adjusted for 4 subplots

			# First subplot: Display the original image
			axes[0].imshow(normalized_img, cmap='gray', alpha=1)
			axes[0].set_title(uploaded_file.name)
			axes[0].set_xticks([])
			axes[0].set_yticks([])

			# Convert canvas to float
			canvas_float = canvas.astype(np.float32)
			# Set the background pixels to NaN
			# Assuming background pixels are labeled as 0
			canvas_float[canvas_float == 0] = np.nan

			# Second subplot: Display the overlay
			axes[1].imshow(normalized_img, cmap='gray', alpha=1)
			axes[1].imshow(canvas_float, cmap=lbl_cmap, alpha=0.5)
			axes[1].set_title('Overlay')
			axes[1].set_xticks([])
			axes[1].set_yticks([])

			# Third subplot: Display the image with bounding boxes
			axes[2].imshow(normalized_img, cmap='gray')
			axes[2].set_title('Detected Organoids')
			axes[2].set_xticks([])
			axes[2].set_yticks([])

			# Add bounding boxes for detected regions in the third subplot
			for region in regionprops(canvas):
				minr, minc, maxr, maxc = region.bbox
				rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr, linewidth=0.5, edgecolor='tab:orange', facecolor='none')
				axes[2].add_patch(rect)

			# Fourth subplot: Display the live/dead classification
			axes[3].imshow(normalized_img, cmap='gray')
			axes[3].set_title('Live/Dead Classification')
			axes[3].set_xticks([])
			axes[3].set_yticks([])

			# Create proxy artists for the legend
			live_patch = mpatches.Patch(color='tab:green', label='Live')
			dead_patch = mpatches.Patch(color='tab:red', label='Dead')

			# Add bounding boxes with color coding for live/dead classification in the fourth subplot
			for region in regionprops(canvas, intensity_image=(255 * normalized_img).astype('uint8')):
				minr, minc, maxr, maxc = region.bbox
				mean_intensity = region.mean_intensity
				color = 'tab:green' if mean_intensity > live_dead_threshold else 'tab:red'
				rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr, linewidth=0.5, edgecolor=color, facecolor='none')
				axes[3].add_patch(rect)

			# Add the legend to the fourth subplot
			axes[3].legend(handles=[live_patch, dead_patch], loc='upper right', ncol=2)

			plt.tight_layout()

			##############

			image_rendering.empty()

			with image_rendering.container():
				st.pyplot(fig, use_container_width=True)
			
			##############
			
			# Save the figure

			# Generate the filename for saving
			filename = f'Cellpose_{uploaded_file.name}'
			filepath = os.path.join(output_dir, filename)

			if os.path.exists(filepath):
				os.remove(filepath)

			# Save the figure
			plt.savefig(filepath, dpi=300, format='tif', bbox_inches='tight')
			# plt.close(fig)

			###########################################################################################

			# Convert the list of results into a single row DataFrame
			df = pd.DataFrame([mask_analysis], columns = ['Image Name', 'Number of Organoids', 'Total Area of Organoids', 'Mean Area of Organoids', 'Live Organoids', 'Dead Organoids', 'Mean Area Live Organoids', 'Mean Area Dead Organoids'])
			
			csv_filename = os.path.join(output_dir, 'Results.csv')

			# Check if the file exists to decide whether to write header
			file_exists = os.path.exists(csv_filename)
			
			# Save the DataFrame as a CSV file in append mode
			# If the file exists, do not write the header again
			df.to_csv(csv_filename, mode='a', index=False, sep=',', header=not file_exists)

			###########################################################################################

			# Generate the filename for saving
			mask_filename = f'Mask_{uploaded_file.name}'
			mask_filepath = os.path.join(output_dir, mask_filename)

			if os.path.exists(mask_filename):
				os.remove(mask_filename)

			# Convert the mask array to an image
			canvas_image = Image.fromarray(canvas)
			canvas_image.save(mask_filepath, format='TIFF')

			###########################################################################################

		status_message1.empty()
		status_message2.empty()
		progress_bar.empty()

		st.success(f'{total_files} images analyzed.')
		st.success(f'Results stored in folder: {output_dir}')
		st.stop()