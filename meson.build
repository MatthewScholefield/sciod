project(
	'sciod',
	'cpp',
	version: '0.1.0',
	default_options : ['cpp_std=c++14'],
	license: 'GPL-3.0'
)

subdir('include')
subdir('src')
subdir('test')

dep = declare_dependency(link_with : lib,
	include_directories : inc)

pkg_mod = import('pkgconfig')
pkg_mod.generate(libraries : lib,
				 version : '0.1',
				 name : 'libsciod',
				 filebase : 'sciod',
				 description : 'A simple neural network library.')

