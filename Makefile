VENV		= .venv
PYTHON		= $(VENV)/bin/python3

DATA		= datasets/data.csv
TRAIN		= datasets/data_train.csv
VALID		= datasets/data_valid.csv

RATIO			?=
SEED			?=
BATCH			?=
LEARNING_RATE	?=
EPOCHS			?=
LAYER			?=
ACTIVATION		?=
MODEL			?=


SPLIT_ARGS	= $(if $(RATIO), --ratio $(RATIO)) \
			  $(if $(SEED), --seed $(SEED))

TRAIN_ARGS	= $(if $(BATCH), --batch $(BATCH)) \
			  $(if $(SEED), --seed $(SEED)) \
			  $(if $(LEARNING_RATE), --learning_rate $(LEARNING_RATE)) \
			  $(if $(EPOCHS), --epochs $(EPOCHS)) \
			  $(if $(LAYER), --layer $(LAYER)) \
			  $(if $(ACTIVATION), --activation $(ACTIVATION)) \
			  $(if $(MODEL), --model $(MODEL))


all: install

$(VENV): requirements.txt
	python3 -m venv $(VENV)
	$(PYTHON) -m pip install -q -r requirements.txt
	touch $(VENV)

install: $(VENV)

split: $(VENV)
	$(PYTHON) split_dataset.py $(DATA) $(SPLIT_ARGS)


train: $(VENV)
	$(PYTHON) train.py $(TRAIN) $(VALID) $(TRAIN_ARGS)

gradient_check: $(VENV)
	$(PYTHON) gradient_check.py

predict: $(VENV)
	$(PYTHON) logreg_predict.py $(TEST) $(WEIGHTS)

clean:
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
	rm -rf $(VALID) $(TRAIN)


fclean: clean
	rm -rf $(VENV)

lint: $(VENV)
	$(PYTHON) -m flake8 --exclude=$(VENV),datasets .

re: fclean all

help:
	@echo "Targets:"
	@echo "  make install         create the virtual environment and install dependencies"
	@echo "  make split           split $(DATA) into a training and a validation set"
	@echo "  make train           train the multilayer perceptron"
	@echo "  make gradient_check  check the backpropagation against numerical gradients"
	@echo "  make predict         evaluate the saved model on a dataset"
	@echo "  make lint            check code style with flake8"
	@echo "  make clean           remove caches and generated datasets"
	@echo "  make fclean          clean + remove the virtual environment"
	@echo "  make re              fclean + install"
	@echo ""
	@echo "Options (passed as VARIABLE=value):"
	@echo "  split   RATIO SEED"
	@echo "  train   BATCH SEED LEARNING_RATE EPOCHS LAYER ACTIVATION MODEL"
	@echo ""
	@echo "Examples:"
	@echo "  make split RATIO=0.75 SEED=42"
	@echo "  make train BATCH=8 EPOCHS=100 LEARNING_RATE=0.03 SEED=42"
	@echo "  make train LAYER=\"24 24 24\"   (quotes needed for several layers)"
	@echo "  make train ACTIVATION=sigmoid BATCH=8"
	@echo "  make train MODEL=./sigmoid_model.json ACTIVATION=sigmoid"
	@echo ""
	@echo "Details of each option:"
	@echo "  make help-split      full help of split_dataset.py"
	@echo "  make help-train      full help of train.py"

help-split: $(VENV)
	@$(PYTHON) split_dataset.py --help

help-train: $(VENV)
	@$(PYTHON) train.py --help

.PHONY: all install lint split train predict clean fclean re \
		help help-split help-train gradient_check
