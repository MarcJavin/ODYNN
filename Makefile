PYTHON = python3

init:
	pip install -r requirements.txt

test:
	coverage run -m unittest discover tests/

install:
	$(PYTHON) setup.py install --user


.PHONY : init test install