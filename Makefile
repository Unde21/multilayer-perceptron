VENV		= .venv
PYTHON		= $(VENV)/bin/python3

TRAIN		= datasets/data.csv
# TEST		= datasets/dataset_test.csv
# WEIGHTS		= weights.json

all: install

$(VENV): requirements.txt
	python3 -m venv $(VENV)
	$(PYTHON) -m pip install -q -r requirements.txt
	touch $(VENV)

install: $(VENV)

train: $(VENV)
	$(PYTHON) logreg_train.py $(TRAIN)

predict: $(VENV)
	$(PYTHON) logreg_predict.py $(TEST) $(WEIGHTS)

evaluate: $(VENV)
	$(PYTHON) evaluate.py $(TRAIN)

clean:
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
	rm -rf plots

fclean: clean
	rm -f $(WEIGHTS) houses.csv
	rm -rf $(VENV)

lint: $(VENV)
	$(PYTHON) -m flake8 --exclude=$(VENV),datasets .

re: fclean all

.PHONY: all install lint train predict evaluate clean fclean re
