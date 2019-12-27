TARGET = venv_init requirements_init
all: $(TARGET)

venv_init: /usr/bin/python3
	python3 -m venv venv

requirements_init: requirements.txt
	venv/bin/pip install -r $<

.PHONY: clean
clean:
	echo "nothing to clean now"
