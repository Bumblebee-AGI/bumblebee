# Optional developer shortcuts (Unix make). On Windows, use equivalent `py -m` commands.

.PHONY: local gateway api worker

ENTITY ?= canary

local:
	py -m bumblebee.main run $(ENTITY)

gateway:
	py -m bumblebee.inference_gateway

api:
	py -m bumblebee.main api --host 0.0.0.0 --port 8080

worker:
	py -m bumblebee.main worker $(ENTITY)
