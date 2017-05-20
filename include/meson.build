headers = [
	'FloatVec.hpp',
	'MetaNet.hpp',
	'NeuralNet.hpp',
	'Node.hpp',
	'Layer.hpp'
]

full_headers = []
foreach header : headers
	full_headers += join_paths('sciod', header)
endforeach
install_headers(full_headers, subdir : 'sciod')

inc = include_directories('.')
