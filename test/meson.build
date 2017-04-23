test_sources = [
	'catch.cpp',
	'simpleTests.cpp'
]

testexe = executable('testexe', test_sources,
					include_directories : inc,
					link_with : lib)

test('sciod test', testexe)

