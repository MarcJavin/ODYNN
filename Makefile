PYTHON = python

init:
	pip install -r requirements.txt

test:
	$(PYTHON) -m unittest discover tests/

install:
	$(PYTHON) setup.py install --user


.PHONY : init test install