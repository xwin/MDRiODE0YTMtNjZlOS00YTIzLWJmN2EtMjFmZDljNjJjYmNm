import pickle
from matplotlib import pyplot as plt

def schema(obj, indent=0):
    pad = "  " * indent
    if isinstance(obj, dict):
        for k, v in obj.items():
            print(f"{pad}{k}: {type(v).__name__}")
            schema(v, indent + 1)
    elif isinstance(obj, list) and obj:
        print(f"{pad}[0]: {type(obj[0]).__name__} (len={len(obj)})")
        schema(obj[0], indent + 1)

# loading ORL dataset
if 0:
	f = open('ORL', 'rb')
	data = pickle.load(f)
	f.close()
	schema(data)
	exit(0)
	for instance in data['train']:
		image_matrix = instance['image']
		image_label = instance['label']
		plt.imshow(image_matrix)
		plt.show()
		print(image_matrix)
		print(image_label)
		# remove the following "break" code if you would like to see more image in the training set
		break
		
	for instance in data['test']:
		image_matrix = instance['image']
		image_label = instance['label']
		plt.imshow(image_matrix)
		plt.show()
		print(image_matrix)
		print(image_label)
		# remove the following "break" code if you would like to see more image in the testing set
		break
		
# loading CIFAR-10 dataset
if 0:
	f = open('CIFAR', 'rb')
	data = pickle.load(f)
	f.close()
	for instance in data['train']:
		image_matrix = instance['image']
		image_label = instance['label']
		plt.imshow(image_matrix)
		plt.show()
		print(image_matrix)
		print(image_label)
		# remove the following "break" code if you would like to see more image in the training set
		break
		
	for instance in data['test']:
		image_matrix = instance['image']
		image_label = instance['label']
		plt.imshow(image_matrix)
		plt.show()
		print(image_matrix)
		print(image_label)
		# remove the following "break" code if you would like to see more image in the testing set
		break

# loading MNIST dataset
if 1:
	f = open('MNIST', 'rb')
	data = pickle.load(f)
	f.close()
	schema(data)

	for instance in data['train']:
		image_matrix = instance['image']
		image_label = instance['label']
		#plt.imshow(image_matrix, cmap='gray')
		#plt.show()
		print(image_matrix.shape, image_matrix.dtype)
		print(image_label)
		# remove the following "break" code if you would like to see more image in the training set
		break
		
	for instance in data['test']:
		image_matrix = instance['image']
		image_label = instance['label']
		#plt.imshow(image_matrix, cmap='gray')
		#plt.show()
		print(image_matrix.shape, image_matrix.dtype)
		print(image_label)
		# remove the following "break" code if you would like to see more image in the testing set
		break
	
	
	