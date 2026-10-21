.PHONY: setup smoke m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 serve test lint release-check

setup:
	python -m pip install -e ".[dev]"

smoke:
	python -m recsys_lab smoke

m1:
	python -m recsys_lab m1

m2:
	python -m recsys_lab m2

m3:
	python -m recsys_lab m3

m4:
	python -m recsys_lab m4

m5:
	python -m recsys_lab m5

m6:
	python -m recsys_lab m6

m7:
	python -m recsys_lab m7

m8:
	python -m recsys_lab m8

m9:
	python -m recsys_lab m9

m10:
	python -m recsys_lab m10

serve:
	python -m recsys_lab serve --host 127.0.0.1 --port 8000

test:
	pytest -q

lint:
	ruff check .

release-check:
	bash scripts/release_check.sh
