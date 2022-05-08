clean:
	cd daw/binary_classification; rm -rf *.so *.c *.html build/ __pycache__; cd -
	cd daw/regression; rm -rf *.so *.c *.html build/ __pycache__; cd -

build:
	cd daw/binary_classification; python3 setup.py build_ext --inplace; cd ..
# 	cd daw/regression; python3 setup.py build_ext --inplace; cd ..

get_deps:
	pip3 install -r requirements.txt

package:
	rm -rf dist/
	python3 setup.py sdist bdist_wheel
	twine check dist/*

upload_pypi_test:
	twine upload --repository-url https://test.pypi.org/legacy/ dist/*

upload_pypi:
	twine upload dist/*

pypi_test: package upload_pypi_test

pypi: package upload_pypi

all: clean get_deps build
