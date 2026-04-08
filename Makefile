# variables:
VENV_DIR = venv
PYTHON = python3
PIP = $(VENV_DIR)/bin/pip
ACTIVATE = . $(VENV_DIR)/bin/activate
REQUIREMENTS = requirements.txt
HEUDICONV_VERSION = 1.3.0
SUBJECTS = $(shell seq -w 1 30)
USER_ID = $(shell id -u)
GROUP_ID = $(shell id -g)

.PHONY: all
all: install

# create virtual environment
.PHONY: venv
venv:
	$(PYTHON) -m venv $(VENV_DIR)

# install dependencies
.PHONY: install
install: venv $(REQUIREMENTS)
	$(PIP) install --upgrade pip
	$(PIP) install -r $(REQUIREMENTS)

# freeze current dependencies to requirements.txt
.PHONY: freeze
freeze: venv
	$(PIP) freeze > $(REQUIREMENTS)

.PHONY: anat
anat:
	@$(foreach sub,$(SUBJECTS), \
		docker run --rm -u $(USER_ID):$(GROUP_ID) -v $(CURDIR)/../highspeed-MEG-raw/data-MRI/:/input:ro -v $(CURDIR):/output:rw -v $(CURDIR)/code:/code:ro \
		nipy/heudiconv:$(HEUDICONV_VERSION) -d /input/MFR{subject}/*IMA \
		-s $(sub) -o /output -f code/heudiconv_heuristic.py -c dcm2niix --bids --overwrite;)

.PHONY: defacing
defacing:
	sh code/defacing.sh $(CURDIR)

.PHONY: bids
bids:
	python code/convert_to_bids.py
