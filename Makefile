init:
	pip3 install -r requirements.txt

test:
	python3 -m unittest discover tests/

install:
    python3 setup.py install --user