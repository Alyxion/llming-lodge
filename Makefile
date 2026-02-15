.PHONY: publish build clean

build:
	python3 scripts/prepare_readme.py
	poetry build

publish: build
	poetry publish
	rm -f README.pypi.md

clean:
	rm -f README.pypi.md
	rm -rf dist/
