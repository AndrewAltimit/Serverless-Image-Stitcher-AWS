try:
    import unzip_requirements
except ImportError:
    pass
	
### NumPy ###
import numpy as np
	
### OpenCV ###
import cv2

import sys, os, json
import boto3

### GLOBAL VARIABLES ###
MIN_MATCH_COUNT        = 10 # Minimum number of keypoint pairs required for image matching	
RATIO_TEST			   = 0.8
REPROJECTION_THRESHOLD = 4.0
H_F_RATIO              = 0.5
		
# Given a directory name and an extension of files to search for,
# the function will return a sorted list of files in the folder.
def get_image_paths_from_folder(dir_name, extension):
	# Store current working directory, then change to desired directory
	cwd = os.getcwd()
	os.chdir(dir_name)
	
	# Get the image paths in the folder with the requested extension
	img_list = os.listdir('./')
	img_list = [dir_name + "/" + name for name in img_list if extension in name.lower() ] 
	img_list.sort()
	
	# Restore the working directory
	os.chdir(cwd)
	
	return img_list
		
		
# Given a list of image paths, return a list of the image data
def read_images(path_list):
	images = []
	for path in path_list:
		image = cv2.imread(path)
		images.append(image)
		
	return images
	
	
# Given an image, return the sift keypoints and descriptors
def get_sift_keypoints(image):
	# Convert image to grayscale
	image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# Initiate SIFT detector
	sift = cv2.xfeatures2d.SIFT_create()

	# find the keypoints and descriptors with SIFT
	keypoints, descriptors = sift.detectAndCompute(image_gray, None)
	
	return keypoints, descriptors
	
	
# given two image heights and widths, and a homography matrix to transform one onto another,
# determine the size of the resulting merge and the offsets which will be applied to the anchor
# such that the anchor is placed in the correct position in the new coordinate system
def get_merge_size(h1, w1, h2, w2, homography):
	# Use perspectiveTransform for calculating size: https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
	points = np.array([[[0,0], [0, h2], [w2, 0], [w2,h2]]]).astype(np.float32)
	top_left, bottom_left, top_right, bottom_right = cv2.perspectiveTransform(points, homography)[0]

	# Get resulting corners
	result_left = int(min(top_left[0], bottom_left[0], 0))
	result_right = int(max(top_right[0], bottom_right[0], w1))
	result_top = int(min(top_left[1], top_right[1], 0))
	result_bottom = int(max(bottom_left[1], bottom_right[1], h1))
	
	# Determine new size
	rows = result_bottom - result_top
	columns = result_right - result_left
	size = (rows, columns, 4) # 4 for alpha layer
	
	#print("LEFT: {}\nRIGHT: {}\nTOP: {}\nBOTTOM: {}\n".format(result_left, result_right, result_top, result_bottom))
	
	# Determine the offset for the anchor image
	# When the anchor is placed on the mosaic, it will be offset by these columns/rows
	row_offset = int(-1* min(top_left[1], top_right[1], 0))
	col_offset = int(-1* min(top_left[0], bottom_left[0], 0))
	offset = (col_offset, row_offset)
   
	return (size, offset)
	

# Given a list of images, return a list of SIFT keypoints
def get_all_keypoints(image_list):
	keypoints = []
	for image in image_list:
		keypoints.append(get_sift_keypoints(image))
	return keypoints
	
# Given two image feature descriptors, find keypoint matches (using FLANN)
def get_keypoint_matches(des1, des2):
	# https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_matcher/py_matcher.html
	index_params = dict(algorithm = 0, trees = 5)
	search_params = dict(checks=50)
	flann = cv2.FlannBasedMatcher(index_params,search_params)
	return flann.knnMatch(des1,des2,k=2)
	
	
# Filter matches based on the ratio test
# https://docs.opencv.org/trunk/dc/dc3/tutorial_py_matcher.html
# Return filtered matches along with the lists of the points for each keypoint pair
def filter_matches_ratio_test(kp1, kp2, all_matches):
	matches = []
	pts1 = []
	pts2 = []
	for m,n in all_matches:
		if m.distance < (RATIO_TEST * n.distance):
			matches.append(m)
			pts1.append(kp1[m.queryIdx].pt)
			pts2.append(kp2[m.trainIdx].pt)
	return matches, pts1, pts2
	

# Filter matches based on fundamental matrix estimation
def filter_via_fundamental_matrix(matches, pts1, pts2):
	# Find the Fundamental Matrix using inlier points
	# http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_epipolar_geometry/py_epipolar_geometry.html
	pts1 = np.int32(pts1)
	pts2 = np.int32(pts2)
	F, fmask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_RANSAC)
	
	#print("Fundamental Matrix:\n", F)
	
	# We select only inlier points that survived the fundamental matrix estimation (identified via the mask)
	pts1 = pts1[fmask.ravel()==1]
	pts2 = pts2[fmask.ravel()==1]
	matches = list(np.asarray(matches)[fmask.ravel() == 1])	
	
	return matches, pts1, pts2
	
	
# Filter matches based on homography matrix estimation	
def filter_via_homography_matrix(matches, pts1, pts2):
	(H, hmask) = cv2.findHomography(pts1, pts2, cv2.RANSAC, REPROJECTION_THRESHOLD)
	
	#print("Homography Matrix:\n", H)
	
	# We select only inlier points that survived the homography matrix estimation
	pts1 = pts1[hmask.ravel()==1]
	pts2 = pts2[hmask.ravel()==1]
	matches = list(np.asarray(matches)[hmask.ravel() == 1])	
	
	return matches, pts1, pts2, H
	
# Given two sets of keypoints/descripters, return the filtered matches, points, H matrix,
# and ratio of points survived from F to H
def get_filtered_matches(kp1, kp2, des1, des2):
	# Get all matches
	all_matches = get_keypoint_matches(des1,des2)
	
	# Filter matches based on the ratio test
	matches, pts1, pts2 = filter_matches_ratio_test(kp1, kp2, all_matches)
	
	# Filter matches based on the Fundamental Matrix
	matches, pts1, pts2 = filter_via_fundamental_matrix(matches, pts1, pts2)
	F_count = len(matches)
	
	# Filter matches based on Homography Matrix
	matches, pts1, pts2, H = filter_via_homography_matrix(matches, pts2, pts1)
	H_count = len(matches)
	
	# Determine how good the match is based on the ratio of matches kept from each filtering step (fundamental to homography)
	ratio = H_count / F_count
	
	return matches, pts1, pts2, H, ratio
	

# Given two images, return the homography matrix
def get_homography(img1,img2):
	keypoints = get_all_keypoints([img1, img2])
	kp1, des1 = keypoints[0]
	kp2, des2 = keypoints[1]
	return get_filtered_matches(kp1, kp2, des1, des2)[3]


	
# Given a list of images and their corresponding keypoints, find the best anchor image and it's pairs of matching images
def find_anchor_image(images, keypoints):
	# Select each image as a potential anchor
	# Go through all other images, see if there exists two other images which pass the test
	anchor_candidates_3 = [] # candidates with 2 other images
	anchor_candidates_2 = [] # candidates with 1 other image
	for i in range(len(images)):	
		# extract keypoints and descriptors for image i
		kp_i, des_i = keypoints[i]	
		good_matches = []
		for j in range(len(keypoints)):
			if i == j:
				continue
				
			# extract keypoints and descriptors for image j
			kp_j, des_j = keypoints[j]
			
			# get matches, points, homography, and match quality ratio (number of points matches survived from the H mask)
			matches, pts1, pts2, H, ratio = get_filtered_matches(kp_i, kp_j, des_i, des_j)
			print("image {} and {} ratio: {:.3f}  matches: {} VALID_MATCH: {}".format(i,j, ratio, len(matches), ratio > H_F_RATIO))
			
			# If the match is high enough quality (good percentage survived from Homography estimation) and it passes the minimum match threshold, it is good
			if ratio >= H_F_RATIO and len(matches) >= MIN_MATCH_COUNT:
				good_matches.append((len(matches), H, j))
			
			# If we have enough good matches for the anchor candidate, then store the results and move onto the next candidate image
			if len(good_matches) == 2:
				anchor_candidates_2.pop(-1)
				# candidate score determined by total number of matches
				score = good_matches[0][0] + good_matches[1][0]
				anchor_candidates_3.append((score, i, good_matches))
				print("Anchor candidate found (3 images):", i)
				break
				
			elif len(good_matches) == 1:
				# candidate score determined by total number of matches
				score = good_matches[0][0]
				anchor_candidates_2.append((score, i, good_matches))
				print("Anchor candidate found (2 images):", i)
			
		print()
			
	
	anchor_candidates_3.sort(reverse = True)
	anchor_candidates_2.sort(reverse = True)
	if len(anchor_candidates_3) > 0:
	
		# image indices
		anchor_ind = anchor_candidates_3[0][1]
		good_matches = anchor_candidates_3[0][2]
		img_1_ind = good_matches[0][2]
		img_2_ind = good_matches[1][2]
		
		# indices array to recover filenames
		indices = [anchor_ind, img_1_ind, img_2_ind]
		
		# Locate image data for these indices
		anchor_image = images[anchor_ind]
		other_images = []
		other_images.append(images[img_1_ind])
		other_images.append(images[img_2_ind])
		
		# Extract out the homography matrices for these matches
		H_matrices = [good_matches[0][1], good_matches[1][1]]
				

		print("Best Anchor: image index {}".format(anchor_ind))

		
	elif len(anchor_candidates_2) > 0:
	
		# image indices
		anchor_ind = anchor_candidates_2[0][1]
		good_matches = anchor_candidates_2[0][2]
		img_1_ind = good_matches[0][2]
		
		# indices array to recover filenames
		indices = [anchor_ind, img_1_ind]
		
		# Locate image data for these indices
		anchor_image = images[anchor_ind]
		other_images = [images[img_1_ind]]
		
		# Extract out the homography matrices for these matches
		H_matrices = [good_matches[0][1]]
				
		print("Best Anchor: image index {}".format(anchor_ind))


	else:
		print("No panorama candidates found. Are you sure it is from the same scene?")
		sys.exit()
		
	return anchor_image, other_images, H_matrices, indices
	
	
# Given two images and the corresponding homography matrix, merge them and return the resulting panorama
# Note: Return image has an alpha layer
def merge_2(base_color, img2_color, H):
	# Convert images to BGRA
	base = cv2.cvtColor(base_color, cv2.COLOR_BGR2BGRA)
	img2 = cv2.cvtColor(img2_color, cv2.COLOR_BGR2BGRA)

	# Determine translation, panorama size, and the offsets for the base image
	h1, w1 = base.shape[:2]
	h2, w2 = img2.shape[:2]
	size, offset = get_merge_size(h1, w1, h2, w2, H)

	(ox, oy) = offset
	translation = np.matrix([[1.0, 0.0, ox],[0, 1.0, oy],[0.0, 0.0, 1.0]])
	H = translation * H
  
	# Determine the overlap
	# For the base image, alpha layer must be factored in
	filter_transparent_ind = np.where(base[:,:,3] == 0)
	base_mask = np.ones_like(base)
	base_mask[filter_transparent_ind] = [0,0,0,0]
	img2_mask = np.ones_like(img2)
	resultmask = np.zeros(size, np.uint8)

	# Create dummy panorama using masks to determine overlap
	cv2.warpPerspective(img2_mask, H, (size[1], size[0]), resultmask, borderMode=cv2.BORDER_TRANSPARENT)
	resultmask[oy:h1+oy, ox:ox+w1] = resultmask[oy:h1+oy, ox:ox+w1] + base_mask
	overlap_ind = np.where((resultmask == [2,2,2,2]).all(axis = 2))

	# Create true panorama now that overlap has been determined
	# Apply warped image onto the resulting panorama
	result = np.zeros_like(resultmask)
	cv2.warpPerspective(img2, H, (size[1], size[0]), result, borderMode=cv2.BORDER_TRANSPARENT)

	# Add in the base image (converted type from uint8 to float so overflow didn't occur)
	result = result.astype(np.float32)
	result[oy:h1+oy, ox:ox+w1] = base + result[oy:h1+oy, ox:ox+w1]
	
	# Overlap indices need to have all color channels divided by 2 (average of the two images)
	result[overlap_ind] = result[overlap_ind] / 2
	result = result.astype(np.uint8)
	
	return result
	
	
# Given an anchor and list of linked images to said anchor
# merge them and return the resulting panorama
def merge_all(anchor_image, other_images, H_matrices, image_names, extension, output_path, output_ID):
	result = merge_2(anchor_image, other_images[0], H_matrices[0])
	output_names = sorted(image_names[0:2])
	if len(other_images) == 2:
		result = merge_2(result, other_images[1], get_homography(result[:, :, :3], other_images[1]))
		cv2.imwrite("{}/Merged-{}.jpg".format(output_path,output_ID), result[:, :, :3])  



# Download a directory from S3
def download_dir(prefix, local, bucket, client, resource):
	keys = []
	dirs = []
	next_token = ''
	base_kwargs = {
		'Bucket':bucket,
		'Prefix':prefix,
	}
	while next_token is not None:
		kwargs = base_kwargs.copy()
		if next_token != '':
			kwargs.update({'ContinuationToken': next_token})
		results = client.list_objects_v2(**kwargs)
		contents = results.get('Contents')
		for i in contents:
			k = i.get('Key')
			if k[-1] != '/':
				keys.append(k)
			else:
				dirs.append(k)
		next_token = results.get('NextContinuationToken')
	for d in dirs:
		dest_pathname = os.path.join(local, d)
		if not os.path.exists(os.path.dirname(dest_pathname)):
			os.makedirs(os.path.dirname(dest_pathname))
	for k in keys:
		dest_pathname = os.path.join(local, k)
		if not os.path.exists(os.path.dirname(dest_pathname)):
			os.makedirs(os.path.dirname(dest_pathname))
		resource.meta.client.download_file(bucket, k, dest_pathname)
			

def stitch(event, context):
	try:
		# Extract Record
		print("Received the following record:\n{}".format(event))
		payload = json.loads(event['Records'][0]['body'])
		sourceFolder = payload["sourceFolder"]
		bucketName = payload["bucketName"]
		destFolder = payload["destFolder"]
		
		
		# Connect to S3 Bucket
		s3_resource = boto3.resource('s3')
		s3_client = boto3.client('s3')
		
		# Download source directory
		download_dir(sourceFolder, '/tmp/image-stitcher/', bucketName, s3_client, s3_resource)

		# Get list of images from the input directory
		directory = '/tmp/image-stitcher/' + sourceFolder
		image_path_list = get_image_paths_from_folder(directory, ".jpg")
		
		# Read in all images and find their corresponding keypoints
		images = read_images(image_path_list)
		keypoints = get_all_keypoints(images)
		
		# Determine an anchor image and its partners (if any)
		anchor_image, other_images, H_matrices, indices = find_anchor_image(images, keypoints)
		
		# Determine output filename based on indices of panorama images
		image_names = []
		extensions = []
		for i in range(len(indices)):
			index = indices[i]
			path = image_path_list[index]
			filename = path.rsplit("/")[-1]
			base_name, extension = filename.rsplit(".")
			image_names.append(base_name)
			extensions.append(extension.lower())
		
		# Merge the images
		merge_all(anchor_image, other_images, H_matrices, image_names, extensions[0], directory, sourceFolder)

		# Upload the result to the destination bucket
		s3_client.upload_file("{}/Merged-{}.jpg".format(directory, sourceFolder), bucketName, "{}/Merged-{}.jpg".format(destFolder, sourceFolder))

		return {
			'statusCode': 200,
			'body': 'Image successfully merged together'
		}
	
	
	except BaseException as error:
		return str(error)	
		