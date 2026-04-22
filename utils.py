import os
import random
import piexif
import shutil

def train_test_split(src_folder, target_root):
	# These are files that should not be used, as the jpgs are corrupted, 
	# and the .db files aren't images
	bad_files_cat = ['Thumbs.db', '666.jpg', '835.jpg']
	bad_files_dog = ['Thumbs.db', '11702.jpg']

	# **ADD YOUR CODE HERE**
	
	#Create Train and Test folders
	categories = ['Cat','Dog']
	for split in ['Train', 'Test']:
		for category in categories:
			path = os.path.join(target_root, split, category)
			os.makedirs(path, exist_ok=True)
	
	#Process cat and dog images
	for category in categories:
		#Get list of all images
		original_dir = os.path.join(src_folder, category)
		all_images = os.listdir(original_dir)

		#filter corrupted
		bad_files = bad_files_cat if category == 'Cat' else bad_files_dog
		valid_images = [img for img in all_images if img not in bad_files]

		#randomize order (for fair split)
		random.shuffle(valid_images)

		#calc 80 split point
		split_point = int(len(valid_images) * 0.8)
		train_list = valid_images[:split_point]
		test_list = valid_images[split_point:]

		#copy images into correct folder
		for img in train_list:
			shutil.copy(os.path.join(original_dir,img), 
			   os.path.join(target_root, 'Train', category, img))
		for img in test_list:
			shutil.copy(os.path.join(original_dir,img), 
			   os.path.join(target_root, 'Test', category, img))	


	# remove corrupted exif data from the dataset
	remove_exif_data(target_root+'Train/')
	remove_exif_data(target_root+'Test/')

# helper function to remove corrupt exif data from Microsoft's dataset
def remove_exif_data(src_folder):
	_, _, cat_images = next(os.walk(src_folder+'Cat/'))
	for img in cat_images:
		try:
			piexif.remove(src_folder+'Cat/'+img)
		except:
			pass

	_, _, dog_images = next(os.walk(src_folder+'Dog/'))
	for img in dog_images:
		try:
			piexif.remove(src_folder+'Dog/'+img)
		except:
			pass


# ** Run train test split on data here **
# At the bottom of utils.py
if __name__ == "__main__":
    train_test_split('PetImages/', 'data/')