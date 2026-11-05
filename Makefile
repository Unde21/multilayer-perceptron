VENV		= .venv
PYTHON		= $(VENV)/bin/python3

DATA		= datasets/data.csv
RATIO		?=
SEED		?=
TRAIN		= datasets/data_train.csv
VALID		= datasets/data_valid.csv
# WEIGHTS		= weights.json

all: install

$(VENV): requirements.txt
	python3 -m venv $(VENV)
	$(PYTHON) -m pip install -q -r requirements.txt
	touch $(VENV)

install: $(VENV)

split: $(VENV)
	$(PYTHON) split_dataset.py $(DATA) $(if $(RATIO), --ratio $(RATIO)) $(if $(SEED), --seed $(SEED))


train: $(VENV)
	$(PYTHON) logreg_train.py $(TRAIN)

predict: $(VENV)
	$(PYTHON) logreg_predict.py $(TEST) $(WEIGHTS)

evaluate: $(VENV)
	$(PYTHON) evaluate.py $(TRAIN)

clean:
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
	rm -rf $(VALID) $(TRAIN)


fclean: clean
	rm -rf $(VENV)

lint: $(VENV)
	$(PYTHON) -m flake8 --exclude=$(VENV),datasets .

re: fclean all

.PHONY: all install lint split train predict evaluate clean fclean re
