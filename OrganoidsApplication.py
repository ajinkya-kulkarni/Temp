import os
import time

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

import sys
# Don't generate the __pycache__ folder locally
sys.dont_write_bytecode = True 
# Print exception without the buit-in python warning
sys.tracebacklimit = 0 

from modules import *

###########################################################################################

model_path = 'OrganoidsTrainedModel'

window_size = 800
overlap = int(0.1 * window_size)

###########################################################################################

with open("logo.jpg", "rb") as f:
	image_data = f.read()

image_bytes = BytesIO(image_data)

st.set_page_config(page_title = 'OrganoIDNet', page_icon = image_bytes, layout = "centered", initial_sidebar_state = "expanded", menu_items = {'About': 'This is a application for demonstrating the OrganoIDNet package. Developed, tested and maintained by Ajinkya Kulkarni: https://github.com/ajinkya-kulkarni at the MPI-NAT, Goettingen. Email: kulkajinkya@gmail.com'})

# Title of the web app

st.title(':blue[Organoid segmentation using OrganoIDNet]')

url1 = "https://doi.org/10.1007/s13402-024-00958-2"
url2 = "https://www.biorxiv.org/content/10.1101/2024.02.12.580032v1"
url3 = "https://github.com/ajinkya-kulkarni/PyOrganoIDNet"
st.markdown("Refer to the following sources for more information about :blue[[OrganoIDNet](%s)], its development and its applicability: :blue[[Article #1](%s)] and :blue[[Article #2](%s)]." % (url3, url1, url2))
st.markdown('Application source code available [here](https://github.com/ajinkya-kulkarni/Temp). Sample image to test this application is available [here](https://github.com/ajinkya-kulkarni/Temp/blob/main/TestImage.tif).')
st.caption('Note that an image of the size of 1500x1000 pixels would take approximately 1-2 minutes to be analyzed.')

with st.form(key = 'form1', clear_on_submit = False):

	# Slider for live/dead threshold
	live_dead_threshold = st.slider("Set the intensity threshold (0-255) for distinguishing live/dead organoids. Default value is 50, as used in :blue[[this article.](%s)]" % url1, min_value=0, max_value=255, value=50, step=1)
	
	live_dead_threshold = int(live_dead_threshold)

	# Multiple file uploader allows user to add their own TIF images
	uploaded_file = st.file_uploader("Drag and drop folder containing the images", type=["tif", "tiff", "png", "jpg"], accept_multiple_files=False)
	
	submitted = st.form_submit_button('Analyze')

	####################################################################################

	if uploaded_file is None:
		st.stop()

	####################################################################################

	if submitted:

		ProgressBar = st.progress(0)
		ProgressBarTime = 0.1

		# Read the uploaded file into a PIL Image object and convert to grayscale

		image = Image.open(uploaded_file).convert('L')
		
		# Convert the image to a numpy array
		image_array = np.array(image)

		normalized_img = read_image_as_grayscale_then_MinMax_normalize(image_array)

		###########################################################################################

		patches, window_coords = patchify(normalized_img, window_size, overlap)
		
		###########################################################################################
		
		predicted_labels = []

		i=1
		for patch in patches:
		
			label = predict_mask_from_image_cellpose(patch, model_path, gpu_usage = True)
		
			smoothed_label = smooth_segmented_labels(label)
		
			predicted_labels.append(smoothed_label)

			time.sleep(ProgressBarTime)
			ProgressBar.progress(float(i/len(patches)))
			i = i+1

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

		mask_analysis = analyze_mask(canvas, image_array, live_dead_threshold, uploaded_file.name)
	
		###########################################################################################

		normalized_img_fig, canvas_float_fig, livedead_fig = create_figures_as_images(normalized_img, canvas, live_dead_threshold)

		image_comparison(img1=normalized_img_fig, img2=canvas_float_fig, label1="Uploaded Image", label2="Segmented Organoids")

		image_comparison(img1=normalized_img_fig, img2=livedead_fig, label1="Uploaded Image", label2="Live Dead Classification")

		###########################################################################################

		# Convert the list of results into a single row DataFrame
		df = pd.DataFrame([mask_analysis], columns = ['Image Name', 'Number of Organoids', 'Total Area of Organoids', 'Mean Area of Organoids', 'Live Organoids', 'Dead Organoids', 'Mean Area Live Organoids', 'Mean Area Dead Organoids'])
		
		# Display the detailed report
		st.markdown("Detailed Report")

		# Show the dataframe
		st.dataframe(df, use_container_width = True, hide_index = True)

		###########################################################################################

		ProgressBar.empty()
