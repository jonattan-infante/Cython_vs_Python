
build:
	python setup.py build_ext --inplace
	echo "[mv] Cambiando de ubicación los ficheros de anotaciones"
	mv src/algorithms_cython/*.html annotations

clean:
	rm -rf build src/algorithms_cython/*.so src/algorithms_cython/*.pyc src/algorithms_cython/*.c