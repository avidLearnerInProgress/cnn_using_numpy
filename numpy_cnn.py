import skimage.data
import numpy
import sys
import matplotlib

def convolution_(img, conv_filter):
	filter_size = conv_filter.shape[1]
	result = numpy.zeros((img.shape))

	for r in numpy.uint16(numpy.arange(filter_size/2.0, img.shape[0] - filter_size/2.0+1)):
		for c in numpy.uint16(numpy.arange(filter_size/2.0, img.shape[1] - filter_size/2.0+1)):
			current_region = img[r-numpy.uint16(numpy.floor(filter_size/2.0)) : r+numpy.uint16(numpy.ceil(filter_size/2.0)), c-numpy.uint16(numpy.floor(filter_size/2.0)) : c+numpy.uint16(numpy.ceil(filter_size/2.0))]			
			current_result = current_region * conv_filter
			conv_sum = numpy.sum(current_result)
			result[r, c] = conv_sum

	final_result = result[numpy.uint16(filter_size/2.0): result.shape[0] - numpy.uint16(filter_size/2.0), numpy.uint16(filter_size/2.0): result.shape[1] - numpy.uint16(filter_size/2.0)]
	return final_result



def convolve(img, conv_filter):
	print("Img shape: "+str(img.shape))
	print("\nConv filter shape: "+str(conv_filter.shape))

	if len(img.shape) > 2 or len(conv_filter.shape) > 3: #edge case
		if img.shape[-1] != conv_filter.shape[-1]:
			print("Error: Number of channels in image and filter must match")
			sys.exit()		

	if conv_filter.shape[1] != conv_filter.shape[2]: 
		print("Error: filter must be of same dimensions")
		sys.exit()		

	if conv_filter.shape[1] % 2 == 0:
		print("Error: dimensions of filter must be odd")
		sys.exit()
	
	output_feature_maps = numpy.zeros((img.shape[0]-conv_filter.shape[1]+1,
									   img.shape[1]-conv_filter.shape[1]+1,
									   conv_filter.shape[0]))
	
	print("Op feature maps shape:"+str(output_feature_maps.shape))
	
	#perform convolution operation
	for filter_num in range(conv_filter.shape[0]):
		print("Filter no: #", filter_num)
		current_filter = conv_filter[filter_num, :] #get current filter we have only 2 filters i.e. 0 and 1
		#print(current_filter)

		#Check if there are mutliple channels for the single filter
		#If so, then each channel will convolve the image
		#conv map stores single feature map
		if len(current_filter.shape) > 2:
			conv_map = convolution_(img[:, :, 0], current_filter[:, :, 0]) #convolve image[0] with filter [0]
			for channel_num in range(1, current_filter.shape[-1]): #convolve remaining image[1] with filter[1], image[2] with filter[2] and so on..
				conv_map += convolution_(img[:, :, channel_num], current_filter[:, :, channel_num])
		else: #there is single channel in filter
			conv_map = convolution_(img, current_filter)
		output_feature_maps[:, :, filter_num]  = conv_map #hold feature map with current filter
		return output_feature_maps

def relu(feature_map):
	relu_out = numpy.zeros(feature_map.shape)
	for map_num in range(feature_map.shape[-1]):
		for r in numpy.arange(0, feature_map.shape[0]):
			for c in numpy.arange(0, feature_map.shape[1]):
				relu_out[r, c , map_num] = numpy.max(feature_map[r, c, map_num], 0)
	return relu_out

def pooling(feature_map, size = 2, stride = 2):
	pool_out = numpy.zeros((numpy.uint16((feature_map.shape[0] - size + 1)/ stride),
							numpy.uint16((feature_map.shape[1] - size + 1)/ stride),
							feature_map.shape[-1]))

	for map_num in range(feature_map.shape[-1]):
		r2 = 0
		for r in numpy.arange(0, feature_map.shape[0] - size - 1, stride):
			c2 = 0
			for c in numpy.arange(0, feature_map.shape[1] - size -1, stride):
				pool_out[r2, c2, map_num]  = numpy.max(feature_map[r : r+size, c : c+size])
				c2 += 1
			r2 += 1
	return pool_out

def main():
	img = skimage.data.chelsea() #load image
	img = skimage.color.rgb2gray(img) #converting image to grayscale

	layer1_filter = numpy.zeros((2,3,3)) #2 filters of 3x3 size each
	
	#assigning 3x3 values for each filter created above i.e. 0 & 1
	layer1_filter[0, :, :] = numpy.array([[[-1, 0, 1],
											[-1, 0, 1],
											[-1, 0, 1]]])

	layer1_filter[1, :, :] = numpy.array([[[1, 1, 1],
										   [0, 0, 0],
										   [-1, -1, -1]]])

	print("--------------Conv Layer-1------------")
	print("Filter for cnn layer-1:")
	#print(layer1_filter)
	
	layer1_feature_map = convolve(img, layer1_filter)
	print("ReLU:")
	layerl_feature_map_relu = relu(layer1_feature_map)
	print(layer1_feature_map_relu)
	print("Pooling:")
	layer1_feature_map_relu_pool = pooling(layerl_feature_map_relu, 2, 2)
	print("---------End of Conv Layer-1---------")


	print("--------------Conv Layer-2------------")
	layer2_filter = numpy.random.rand(3, 5, 5, layer1_feature_map_relu_pool.shape[-1])
	print("Filter for cnn layer-2:")
	#print(layer2_filter)
	layer2_feature_map = convolve(layer1_feature_map_relu_pool, layer2_filter)
	print("ReLU:")
	layer2_feature_map_relu = relu(layer2_feature_map)
	print("Pooling:")
	layer2_feature_map_relu_pool = pooling(layer2_feature_map_relu, 2, 2)
	print("---------End of Conv Layer-2---------")
	

	print("--------------Conv Layer-3------------")
	layer3_filter = numpy.random.rand(1, 7, 7, layer2_feature_map_relu_pool.shape[-1])
	print("Filter for cnn layer-3:")
	#print(layer3_filter)
	
	layer3_feature_map = convolve(layer2_feature_map_relu_pool, layer3_filter)
	print("ReLU:")
	layer3_feature_map_relu = relu(layer3_feature_map)
	print("Pooling:")
	layer3_feature_map_relu_pool = pooling(layer3_feature_map_relu, 2, 2)
	print("---------End of Conv Layer-3---------")
	

	# Graphing results
	fig0, ax0 = matplotlib.pyplot.subplots(nrows=1, ncols=1)
	ax0.imshow(img).set_cmap("gray")
	ax0.set_title("Input Image")
	ax0.get_xaxis().set_ticks([])
	ax0.get_yaxis().set_ticks([])
	matplotlib.pyplot.savefig("in_img.png", bbox_inches="tight")
	matplotlib.pyplot.close(fig0)

	# Layer 1
	fig1, ax1 = matplotlib.pyplot.subplots(nrows=3, ncols=2)
	ax1[0, 0].imshow(layer1_feature_map[:, :, 0]).set_cmap("gray")
	ax1[0, 0].get_xaxis().set_ticks([])
	ax1[0, 0].get_yaxis().set_ticks([])
	ax1[0, 0].set_title("L1-Map1")

	ax1[0, 1].imshow(layer1_feature_map[:, :, 1]).set_cmap("gray")
	ax1[0, 1].get_xaxis().set_ticks([])
	ax1[0, 1].get_yaxis().set_ticks([])
	ax1[0, 1].set_title("L1-Map2")

	ax1[1, 0].imshow(layer1_feature_map_relu[:, :, 0]).set_cmap("gray")
	ax1[1, 0].get_xaxis().set_ticks([])
	ax1[1, 0].get_yaxis().set_ticks([])
	ax1[1, 0].set_title("L1-Map1ReLU")

	ax1[1, 1].imshow(layer1_feature_map_relu[:, :, 1]).set_cmap("gray")
	ax1[1, 1].get_xaxis().set_ticks([])
	ax1[1, 1].get_yaxis().set_ticks([])
	ax1[1, 1].set_title("L1-Map2ReLU")

	ax1[2, 0].imshow(layer1_feature_map_relu_pool[:, :, 0]).set_cmap("gray")
	ax1[2, 0].get_xaxis().set_ticks([])
	ax1[2, 0].get_yaxis().set_ticks([])
	ax1[2, 0].set_title("L1-Map1ReLUPool")

	ax1[2, 1].imshow(layer1_feature_map_relu_pool[:, :, 1]).set_cmap("gray")
	ax1[2, 0].get_xaxis().set_ticks([])
	ax1[2, 0].get_yaxis().set_ticks([])
	ax1[2, 1].set_title("L1-Map2ReLUPool")

	matplotlib.pyplot.savefig("L1.png", bbox_inches="tight")
	matplotlib.pyplot.close(fig1)

	# Layer 2
	fig2, ax2 = matplotlib.pyplot.subplots(nrows=3, ncols=3)
	ax2[0, 0].imshow(layer2_feature_map[:, :, 0]).set_cmap("gray")
	ax2[0, 0].get_xaxis().set_ticks([])
	ax2[0, 0].get_yaxis().set_ticks([])
	ax2[0, 0].set_title("L2-Map1")

	ax2[0, 1].imshow(layer2_feature_map[:, :, 1]).set_cmap("gray")
	ax2[0, 1].get_xaxis().set_ticks([])
	ax2[0, 1].get_yaxis().set_ticks([])
	ax2[0, 1].set_title("L2-Map2")

	ax2[0, 2].imshow(layer2_feature_map[:, :, 2]).set_cmap("gray")
	ax2[0, 2].get_xaxis().set_ticks([])
	ax2[0, 2].get_yaxis().set_ticks([])
	ax2[0, 2].set_title("L2-Map3")

	ax2[1, 0].imshow(layer2_feature_map_relu[:, :, 0]).set_cmap("gray")
	ax2[1, 0].get_xaxis().set_ticks([])
	ax2[1, 0].get_yaxis().set_ticks([])
	ax2[1, 0].set_title("L2-Map1ReLU")

	ax2[1, 1].imshow(layer2_feature_map_relu[:, :, 1]).set_cmap("gray")
	ax2[1, 1].get_xaxis().set_ticks([])
	ax2[1, 1].get_yaxis().set_ticks([])
	ax2[1, 1].set_title("L2-Map2ReLU")

	ax2[1, 2].imshow(layer2_feature_map_relu[:, :, 2]).set_cmap("gray")
	ax2[1, 2].get_xaxis().set_ticks([])
	ax2[1, 2].get_yaxis().set_ticks([])
	ax2[1, 2].set_title("L2-Map3ReLU")

	ax2[2, 0].imshow(layer2_feature_map_relu_pool[:, :, 0]).set_cmap("gray")
	ax2[2, 0].get_xaxis().set_ticks([])
	ax2[2, 0].get_yaxis().set_ticks([])
	ax2[2, 0].set_title("L2-Map1ReLUPool")

	ax2[2, 1].imshow(layer2_feature_map_relu_pool[:, :, 1]).set_cmap("gray")
	ax2[2, 1].get_xaxis().set_ticks([])
	ax2[2, 1].get_yaxis().set_ticks([])
	ax2[2, 1].set_title("L2-Map2ReLUPool")

	ax2[2, 2].imshow(layer2_feature_map_relu_pool[:, :, 2]).set_cmap("gray")
	ax2[2, 2].get_xaxis().set_ticks([])
	ax2[2, 2].get_yaxis().set_ticks([])
	ax2[2, 2].set_title("L2-Map3ReLUPool")

	matplotlib.pyplot.savefig("L2.png", bbox_inches="tight")
	matplotlib.pyplot.close(fig2)

	# Layer 3
	fig3, ax3 = matplotlib.pyplot.subplots(nrows=1, ncols=3)
	ax3[0].imshow(layer3_feature_map[:, :, 0]).set_cmap("gray")
	ax3[0].get_xaxis().set_ticks([])
	ax3[0].get_yaxis().set_ticks([])
	ax3[0].set_title("L3-Map1")

	ax3[1].imshow(layer3_feature_map_relu[:, :, 0]).set_cmap("gray")
	ax3[1].get_xaxis().set_ticks([])
	ax3[1].get_yaxis().set_ticks([])
	ax3[1].set_title("L3-Map1ReLU")

	ax3[2].imshow(layer3_feature_map_relu_pool[:, :, 0]).set_cmap("gray")
	ax3[2].get_xaxis().set_ticks([])
	ax3[2].get_yaxis().set_ticks([])
	ax3[2].set_title("L3-Map1ReLUPool")

	matplotlib.pyplot.savefig("L3.png", bbox_inches="tight")

if __name__ == '__main__':
	main()

