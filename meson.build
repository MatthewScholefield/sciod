project(
	'sciod',
	'cpp',
	version: '0.1.0',
	default_options : ['cpp_std=c++14'],
	license: 'GPL-3.0'
)

inc = include_directories('include')

subdir('include')
subdir('src')
subdir('test')

pkg_mod = import('pkgconfig')
pkg_mod.generate(libraries : sciodlib,
				 version : '0.1',
				 name : 'libsciod',
				 filebase : 'sciod',
				 description : 'A simple neural network library.')

