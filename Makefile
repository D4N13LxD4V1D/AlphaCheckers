install: Pipfile
	python3 -m pipenv install --skip-lock

board:
	python3 -m pipenv run tensorboard --logdir Graph

run:
	python3 -m pipenv run python3 run.py

clean:
	python3 -m pipenv --rm

all: install kernel run 
