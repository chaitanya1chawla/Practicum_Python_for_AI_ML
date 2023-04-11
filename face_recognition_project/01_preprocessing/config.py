# Paths
# put a "/" at the end of path
image_dir = 'C:/Users/katha/Downloads/Images_Pyeda/'  # set image directory for host

# Data variables local pictures
im_height = 50  # from lfw dataset
im_width = 37  # from lfw dataset
class_dict = {'kat': 0, 'cha': 1, 'other': 2}
# augment_modes = []    # add modes if needed
augment_modes = ['rotate', 'flip', 'noise', 'contrast']

# # LFW Picture variables
# min_faces = 20
# resize = 0.4

# Hyper parameters
split_data = False  # shuffle and split data in train, test, val
