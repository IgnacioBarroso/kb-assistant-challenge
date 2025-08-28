.PHONY: devcontainer-build

devcontainer-build:
	[ -e .env ] || touch .env
	docker compose -f .devcontainer/docker-compose.yml build kbac-devcontainer

run-server:
	docker compose up -d
