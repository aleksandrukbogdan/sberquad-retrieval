.PHONY: eval tune test mini extra

eval:
	python -m lab.eval --split dev

tune:
	python -m lab.eval --split tune

test:
	pytest -q

mini:
	python -m lab.eval --split mini --check

extra:
	python -m lab.eval --split extra
