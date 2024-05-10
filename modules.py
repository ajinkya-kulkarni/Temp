import io
import numpy as np
import cv2
from PIL import Image
from skimage.measure import regionprops
from matplotlib import colors
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import streamlit as st
from cellpose import models

######################################################################################

def read_image_as_grayscale_then_MinMax_normalize(image_array):
	"""
	Read an image from a given path, convert it to grayscale, and apply Min-Max normalization.

	Parameters:
	image_array (str): The image_array to be processed.

	Returns:
	numpy.ndarray: A normalized grayscale image represented as a 2D array with values ranging from 0 to 1.

	Raises:
	ValueError: If the image has no variance (e.g., it might be empty or corrupted).
	"""

	# Convert the image into a NumPy array for easier manipulation
	img_array = np.asarray(image_array, dtype=np.float32)
	
	# Find the minimum and maximum pixel values in the image
	min_val = img_array.min()
	max_val = img_array.max()
	
	# Perform Min-Max normalization if the image has more than one unique pixel value
	if max_val - min_val > 0:
		normalized_img = (img_array - min_val) / (max_val - min_val)
	else:
		# If the image has no variance, raise an error
		raise ValueError('Image has no variance, it might be empty or corrupted')
	
	return normalized_img

######################################################################################

# generate random colors for instances

def random_label_cmap(labels):
	n_labels = len(np.unique(labels)) - 1
	cmap = [[0, 0, 0]] + np.random.rand(n_labels, 3).tolist()
	cmap = colors.ListedColormap(cmap)
	return cmap

######################################################################################

def smooth_segmented_labels(image):
	"""
	Smooths the segmented labels in an image using convex hull and replaces each label with its smoothed version.

	Parameters:
		image (numpy.ndarray): The input segmented image array (grayscale).

	Returns:
		numpy.ndarray: The image array with smoothed labels, as uint16.
	"""
	# Ensure the image is in the correct format
	if len(image.shape) != 2:
		raise ValueError("Input image must be a grayscale image")

	# Initialize the output image
	smoothed_image = np.zeros_like(image)

	# Unique labels in the image (excluding background)
	unique_labels = np.unique(image)
	unique_labels = unique_labels[unique_labels != 0]

	for label in unique_labels:
		# Create a binary mask for the current label
		label_mask = (image == label).astype(np.uint8)

		# Find contours for this label
		contours, _ = cv2.findContours(label_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

		# Calculate convex hull for each contour and draw it in the output image
		for contour in contours:
			hull = cv2.convexHull(contour)
			cv2.drawContours(smoothed_image, [hull], -1, int(label), thickness=cv2.FILLED)

	return smoothed_image.astype(np.uint16)  # Ensure output is uint16

######################################################################################

def compile_label_info(predicted_labels, window_coords, min_area_threshold=10):
	"""
	Compiles information about each label in the predicted labels of image patches.

	Parameters:
	predicted_labels (list of ndarray): A list of 2D arrays where each element is a labeled image patch.
	window_coords (list of tuples): Coordinates of each patch in the format (x_offset, y_offset, width, height).
	min_area_threshold (int): Minimum area threshold for regions to be considered. Default is 20.

	Returns:
	list: A list of dictionaries, where each dictionary contains information about a label.

	The function iterates over each patch and its labels, calculates the global position of each label,
	and compiles information like global centroid, global bounding box, and the binary image of the label.
	"""

	all_labels_info = []
	new_label_id = 1  # Initialize label ID counter

	# Iterate over each patch and its corresponding window coordinates
	for patch_labels, (x_offset, y_offset, _, _) in zip(predicted_labels, window_coords):
		# Iterate over each region in the patch
		for region in regionprops(patch_labels):
			area = region.area

			# Filter out regions smaller than the minimum area threshold
			if area > min_area_threshold:
				# Calculate global centroid coordinates
				local_centroid_y, local_centroid_x = region.centroid
				global_center_x = local_centroid_x + x_offset
				global_center_y = local_centroid_y + y_offset

				# Calculate global bounding box coordinates
				minr, minc, maxr, maxc = region.bbox
				global_bbox = (minc + x_offset, minr + y_offset, maxc + x_offset, maxr + y_offset)

				# Extract the binary image of the label within its local bounding box
				label_image = new_label_id * region.image.astype(int)

				# Compile label information in a dictionary
				label_info = {
					'label_id': new_label_id,
					'global_centroid': (global_center_x, global_center_y),
					'global_bbox': global_bbox,
					'label_image': label_image
				}
				all_labels_info.append(label_info)

				new_label_id += 1  # Increment label ID for the next entry

	return all_labels_info

######################################################################################

def patchify(image, window_size, overlap):
	"""
	Divides an image into overlapping or non-overlapping patches.

	Parameters:
	image (ndarray): The input image as a 2D array.
	window_size (int): The size of each square patch (in pixels).
	overlap (int): The amount of overlap between adjacent patches (in pixels).

	Returns:
	tuple: A tuple containing two elements:
		- An array of image patches.
		- A list of coordinates for each patch in the format (x_start, y_start, x_end, y_end).

	The function calculates the stride (step size) based on the window size and overlap,
	then iterates over the image to extract patches and their corresponding coordinates.
	"""

	height, width = image.shape
	stride = window_size - overlap  # Calculate stride (step size) based on window size and overlap

	# Generate start coordinates for patches
	x_coords = list(range(0, width - window_size + 1, stride))
	y_coords = list(range(0, height - window_size + 1, stride))

	# Add an extra coordinate if the last patch doesn't align perfectly with the image edge
	if x_coords[-1] != width - window_size:
		x_coords.append(width - window_size)
	if y_coords[-1] != height - window_size:
		y_coords.append(height - window_size)

	windows = []  # To store image patches
	window_coords = []  # To store coordinates of each patch

	# Iterate over y and x coordinates to extract patches
	for y in y_coords:
		for x in x_coords:
			window = image[y:y + window_size, x:x + window_size]  # Extract patch
			windows.append(window)
			window_coords.append((x, y, x + window_size, y + window_size))  # Store patch coordinates

	windows = np.asarray(windows)  # Convert list of windows to numpy array for efficient processing

	return windows, window_coords

######################################################################################

def remove_border_labels(label_image, patch_coords, original_image, neutral_value=0):
	"""
	Remove labels at the borders of the patch, unless the border is shared with the original image.

	Parameters:
	- label_image: An array where each connected region is assigned a unique integer label.
	- patch_coords: A tuple (x1, y1, x2, y2) indicating the coordinates of the patch within the original image.
	- image_shape: The shape (height, width) of the original image.
	- neutral_value: The value to assign to removed labels.

	Returns:
	- An array of the same shape as `label_image` with the appropriate border labels removed.
	"""
	x1, y1, x2, y2 = patch_coords
	height, width = original_image.shape
	border_labels = set()

	# Check each border and add labels to remove if not at the edge of the original image
	if y1 != 0:
		border_labels.update(label_image[0, :])
	if y2 != height:
		border_labels.update(label_image[-1, :])
	if x1 != 0:
		border_labels.update(label_image[:, 0])
	if x2 != width:
		border_labels.update(label_image[:, -1])

	# Create a mask of regions to remove
	border_mask = np.isin(label_image, list(border_labels))

	# Set the border labels to neutral_value (background)
	cleaned_label_image = label_image.copy()
	cleaned_label_image[border_mask] = neutral_value

	return cleaned_label_image

######################################################################################

def non_maximum_suppression(boxes, overlapThresh):
	"""
	Performs non-maximum suppression to reduce overlapping bounding boxes.

	Parameters:
	boxes (list of tuples): A list of bounding boxes, each represented as a tuple (x1, y1, x2, y2).
	overlapThresh (float): The overlap threshold for suppression. Boxes with overlap greater than this threshold are suppressed.

	Returns:
	list: A list of indices of boxes that have been selected after suppression.

	This function sorts the boxes by their bottom-right y-coordinate, selects the box with the largest y-coordinate,
	and suppresses boxes that overlap significantly with this selected box.
	"""

	if len(boxes) == 0:
		return []

	selected_indices = []  # Initialize the list of selected indices

	# Sort the boxes by the bottom-right y-coordinate (y2)
	sorted_indices = np.argsort([box[3] for box in boxes])

	while len(sorted_indices) > 0:
		# Select the box with the largest y2 and remove it from the list
		current_index = sorted_indices[-1]
		selected_indices.append(current_index)
		sorted_indices = sorted_indices[:-1]

		# Iterate over the remaining boxes and remove those that overlap significantly
		for other_index in sorted_indices.copy():
			if does_overlap(boxes[current_index], boxes[other_index], overlapThresh):
				sorted_indices = sorted_indices[sorted_indices != other_index]

	return selected_indices

######################################################################################

def does_overlap(box1, box2, overlapThresh):
	"""
	Determines if two boxes overlap more than a given threshold.

	Parameters:
	box1 (tuple): The first bounding box (x1, y1, x2, y2).
	box2 (tuple): The second bounding box (x1, y1, x2, y2).
	overlapThresh (float): Overlap threshold for determining significant overlap.

	Returns:
	bool: True if the overlap is greater than the threshold, False otherwise.

	This function calculates the intersection area of two boxes and compares it against
	the overlap threshold relative to the smaller box area.
	"""

	# Calculate the intersection of two boxes
	x_min = max(box1[0], box2[0])
	y_min = max(box1[1], box2[1])
	x_max = min(box1[2], box2[2])
	y_max = min(box1[3], box2[3])

	# Calculate area of intersection
	intersection_area = max(0, x_max - x_min) * max(0, y_max - y_min)

	# Calculate area of both boxes
	box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
	box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

	# Check if the intersection is greater than the threshold
	return intersection_area > overlapThresh * min(box1_area, box2_area)

######################################################################################

def place_labels_on_canvas(normalized_img, nms_region_info_list):
	"""
	Places labels on a canvas with random decisions for overlapping areas.

	Parameters:
	normalized_img (numpy.ndarray): The base image used to determine the canvas size.
	nms_region_info_list (list): List of dictionaries containing label info and bounding box.

	Returns:
	numpy.ndarray: The canvas with labels placed on it.
	"""
	canvas_height, canvas_width = normalized_img.shape
	canvas = np.zeros((canvas_height, canvas_width), dtype=np.int16)

	for label_info in nms_region_info_list:
		bbox = label_info['global_bbox']
		label_img = label_info['label_image']

		start_x, start_y, end_x, end_y = bbox
		height, width = label_img.shape[:2]

		temp_canvas = np.zeros_like(canvas)
		temp_canvas[start_y:start_y + height, start_x:start_x + width] = label_img

		overlap_area = (canvas > 0) & (temp_canvas > 0)

		for y in range(start_y, start_y + height):
			for x in range(start_x, start_x + width):
				if overlap_area[y, x]:
					if np.random.rand() < 0.5:  # 50% chance
						temp_canvas[y, x] = canvas[y, x]

		canvas[temp_canvas > 0] = temp_canvas[temp_canvas > 0]

	return canvas

######################################################################################

def predict_mask_from_image_cellpose(normalized_img_patch, model_path, gpu_usage = False):

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

def analyze_mask(mask, original_img, live_dead_threshold, filename):

	# Ensure mask is uint16
	mask = mask.astype(np.uint16)

	# Calculate properties of each labeled region
	props = regionprops(mask, intensity_image=original_img)

	# Number of labels
	num_labels = len(props)

	# Total area of all labels
	total_area = sum([prop.area for prop in props])

	# Overall mean area
	mean_area = np.mean([prop.area for prop in props]) if num_labels > 0 else 0
	mean_area = round(mean_area, 2)

	# Live/dead classification
	live_labels = [prop for prop in props if prop.mean_intensity > live_dead_threshold]
	dead_labels = [prop for prop in props if prop.mean_intensity <= live_dead_threshold]

	# Total area of live and dead labels
	total_live_area = sum([prop.area for prop in live_labels])
	total_dead_area = sum([prop.area for prop in dead_labels])

	no_live_organoids = len(live_labels)
	no_dead_organoids = len(dead_labels)

	mean_area_live_organoids = total_live_area / no_live_organoids if no_live_organoids > 0 else 0
	mean_area_live_organoids = round(mean_area_live_organoids, 2)

	mean_area_dead_organoids = total_dead_area / no_dead_organoids if no_dead_organoids > 0 else 0
	mean_area_dead_organoids = round(mean_area_dead_organoids, 2)

	# Overall mean area
	mean_area = np.mean([prop.area for prop in props]) if num_labels > 0 else 0
	mean_area = round(mean_area, 2)

	result = [filename, num_labels, total_area, mean_area, no_live_organoids, no_dead_organoids, mean_area_live_organoids, mean_area_dead_organoids]

	return result

###########################################################################################

# Function to convert a Matplotlib figure to a PIL Image
def figure_to_image(fig):
	"""Convert a Matplotlib figure to a PIL Image and return it"""
	buf = io.BytesIO()
	fig.savefig(buf, format='png', bbox_inches='tight')
	buf.seek(0)
	img = Image.open(buf)
	plt.close(fig)  # Close the figure to free memory
	return img

# Function to create the figures and convert them to PIL Images
def create_figures_as_images(normalized_img, canvas, live_dead_threshold):
	# Convert canvas to float and set background pixels to NaN
	canvas_float = canvas.astype(np.float32)
	canvas_float[canvas_float == 0] = np.nan

	# First figure: Display the original image
	fig1, ax1 = plt.subplots(figsize=(10, 10))
	ax1.imshow(normalized_img, cmap='gray', alpha=1)
	ax1.set_xticks([])
	ax1.set_yticks([])
	img1 = figure_to_image(fig1)

	# Second figure: Display the overlay
	fig2, ax2 = plt.subplots(figsize=(10, 10))
	ax2.imshow(normalized_img, cmap='gray', alpha=1)
	ax2.imshow(canvas_float, cmap=random_label_cmap(canvas_float), alpha=0.5)  # Make sure random_label_cmap is defined or replaced
	ax2.set_xticks([])
	ax2.set_yticks([])
	img2 = figure_to_image(fig2)

	# Third figure: Display the live/dead classification
	fig3, ax3 = plt.subplots(figsize=(10, 10))
	ax3.imshow(normalized_img, cmap='gray')
	ax3.set_xticks([])
	ax3.set_yticks([])
	live_patch = mpatches.Patch(color='tab:green', label='Live')
	dead_patch = mpatches.Patch(color='tab:red', label='Dead')
	for region in regionprops(canvas, intensity_image=(255 * normalized_img).astype('uint16')):
		minr, minc, maxr, maxc = region.bbox
		mean_intensity = region.mean_intensity
		color = 'tab:green' if mean_intensity > live_dead_threshold else 'tab:red'
		rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr, linewidth=1, edgecolor=color, facecolor='None')
		ax3.add_patch(rect)
	img3 = figure_to_image(fig3)

	return img1, img2, img3

###########################################################################################

import streamlit.components.v1 as components
import base64
from typing import Union, Tuple
import requests

def read_image_and_convert_to_base64(image: Union[Image.Image, str, np.ndarray]) -> Tuple[str, int, int]:
	"""
	Reads an image in PIL Image, file path, or numpy array format and returns a base64-encoded string of the image
	in JPEG format, along with its width and height.

	Args:
		image: An image in PIL Image, file path, or numpy array format.

	Returns:
		A tuple containing:
		- base64_src (str): A base64-encoded string of the image in JPEG format.
		- width (int): The width of the image in pixels.
		- height (int): The height of the image in pixels.

	Raises:
		TypeError: If the input image is not of a recognized type.

	Assumes:
		This function assumes that the input image is a valid image in PIL Image, file path, or numpy array format.
		It also assumes that the necessary libraries such as Pillow and scikit-image are installed.

	"""
	# Set the maximum image size to None to allow reading of large images
	Image.MAX_IMAGE_PIXELS = None

	# If input image is PIL Image, convert it to RGB format
	if isinstance(image, Image.Image):
		image_pil = image.convert('RGB')

	# If input image is a file path, open it using requests library if it's a URL, otherwise use PIL Image's open function
	elif isinstance(image, str):
		try:
			image_pil = Image.open(
				requests.get(image, stream=True).raw if str(image).startswith("http") else image
			).convert("RGB")
		except:
			# If opening image using requests library fails, try to use scikit-image library to read the image
			try:
				import skimage.io
			except ImportError:
				raise ImportError("Please run 'pip install -U scikit-image imagecodecs' for large image handling.")

			# Read the image using scikit-image and convert it to a PIL Image
			image_sk = skimage.io.imread(image).astype(np.uint8)
			if len(image_sk.shape) == 2:
				image_pil = Image.fromarray(image_sk, mode="1").convert("RGB")
			elif image_sk.shape[2] == 4:
				image_pil = Image.fromarray(image_sk, mode="RGBA").convert("RGB")
			elif image_sk.shape[2] == 3:
				image_pil = Image.fromarray(image_sk, mode="RGB")
			else:
				raise TypeError(f"image with shape: {image_sk.shape[3]} is not supported.")

	# If input image is a numpy array, create a PIL Image from it
	elif isinstance(image, np.ndarray):
		if image.shape[0] < 5:
			image = image[:, :, ::-1]
		image_pil = Image.fromarray(image).convert("RGB")

	# If input image is not of a recognized type, raise a TypeError
	else:
		raise TypeError("read image with 'pillow' using 'Image.open()'")

	# Get the width and height of the image
	width, height = image_pil.size

	# Save the PIL Image as a JPEG image with maximum quality (100) and no subsampling
	in_mem_file = io.BytesIO()
	image_pil.save(in_mem_file, format="JPEG", subsampling=0, quality=100)

	# Encode the bytes of the JPEG image in base64 format
	img_bytes = in_mem_file.getvalue()
	image_str = base64.b64encode(img_bytes).decode("utf-8")

	# Create a base64-encoded string of the image in JPEG format
	base64_src = f"data:image/jpg;base64,{image_str}"

	# Return the base64-encoded string along with the width and height of the image
	return base64_src, width, height

######################################################

def image_comparison(
	img1: str,
	img2: str,
	label1: str,
	label2: str,
	width_value = 674,
	show_labels: bool=True,
	starting_position: int=50,
) -> components.html:
	"""
	Creates an HTML block containing an image comparison slider of two images.

	Args:
		img1 (str): A string representing the path or URL of the first image to be compared.
		img2 (str): A string representing the path or URL of the second image to be compared.
		label1 (str): A label to be displayed above the first image in the slider.
		label2 (str): A label to be displayed above the second image in the slider.
		width_value (int, optional): The maximum width of the slider in pixels. Defaults to 500.
		show_labels (bool, optional): Whether to show the labels above the images in the slider. Defaults to True.
		starting_position (int, optional): The starting position of the slider. Defaults to 50.

	Returns:
		A Dash HTML component that displays an image comparison slider.

	"""
		# Convert the input images to base64 format
	img1_base64, img1_width, img1_height = read_image_and_convert_to_base64(img1)
	img2_base64, img2_width, img2_height = read_image_and_convert_to_base64(img2)

	# Get the maximum width and height of the input images
	img_width = int(max(img1_width, img2_width))
	img_height = int(max(img1_height, img2_height))

	# Calculate the aspect ratio of the images
	h_to_w = img_height / img_width

	# Determine the height of the slider based on the width and aspect ratio
	if img_width < width_value:
		width = img_width
	else:
		width = width_value
	height = int(width * h_to_w)

	# Load CSS and JS for the slider
	cdn_path = "https://cdn.knightlab.com/libs/juxtapose/latest"
	css_block = f'<link rel="stylesheet" href="{cdn_path}/css/juxtapose.css">'
	js_block = f'<script src="{cdn_path}/js/juxtapose.min.js"></script>'

	# Create the HTML code for the slider
	htmlcode = f"""
		<style>body {{ margin: unset; }}</style>
		{css_block}
		{js_block}
		<div id="foo" style="height: {height}; width: {width};"></div>
		<script>
		slider = new juxtapose.JXSlider('#foo',
			[
				{{
					src: '{img1_base64}',
					label: '{label1}',
				}},
				{{
					src: '{img2_base64}',
					label: '{label2}',
				}}
			],
			{{
				animate: true,
				showLabels: {str(show_labels).lower()},
				showCredits: true,
				startingPosition: "{starting_position}%",
				makeResponsive: true,
			}});
		</script>
		"""

	# Create a Dash HTML component from the HTML code
	static_component = components.html(htmlcode, height=height, width=width)

	return static_component

##########################################################################