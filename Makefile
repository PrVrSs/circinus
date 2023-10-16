SHELL := /usr/bin/env bash


.PONY: unit
unit:
	poetry run pytest


.PONY: lint
lint:
	poetry run pylint circinus


.PONY: mypy
mypy:
	poetry run mypy circinus


test: lint unit