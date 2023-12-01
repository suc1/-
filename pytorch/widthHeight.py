def convolution(_input:int, filter:int, stride=1, padding=0) -> int:
	# 卷积后的维度: `height = (input - filter + 2 * padding) / stride + 1`
	return (_input - filter + 2 * padding) // stride + 1
	
	
def pool(_input:int, _pool:int, stride=1) -> int:
	# 池化层后的维度: `height = [(input - pool) / stride] + 1`
	return (_input - _pool) // stride + 1

# 224 * 224 * 3
convolution(224, 11, 4)     # 54 * 54 * 96 
pool(54, 3, 2)              # 26 * 26 * 96

convolution(26, 5, 1, 2)    # 26 * 26 * 256
pool(26, 3, 2)              # 12 * 12 * 256

convolution(12, 3, 1, 1)    # 12 * 12 * 384
convolution(12, 3, 1, 1)    # 12 * 12 * 384
convolution(12, 3, 1, 1)    # 12 * 12 * 256

pool(12, 3, 2)              # 5 * 5 * 256 = 6400