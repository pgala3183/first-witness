# First Witness. On Windows without make, run the commands under each target.
PY ?= python
.PHONY: help api web feeds check bench up

help:
	@echo "make api     # API + live vision + call agent on :8080 (reads .env)"
	@echo "make web     # security console on :8090; open http://127.0.0.1:8090/?ws=ws://127.0.0.1:8080/ws"
	@echo "make feeds   # rebuild the three camera feeds + timing data from data/clips"
	@echo "make check   # every self-check: brain, api, voice, call model, briefing, vision, web"
	@echo "make bench   # call-agent benchmark table from finetune/results"
	@echo "make up      # docker compose up api"

api:
	PYTHONPATH=. $(PY) -m services.api --host 127.0.0.1 --port 8080

web:
	$(PY) -m http.server 8090 --bind 127.0.0.1 --directory web

feeds:
	$(PY) scripts/build_demo_from_clips.py

check:
	PYTHONPATH=. $(PY) services/brain/check.py
	PYTHONPATH=. $(PY) services/api/check.py
	PYTHONPATH=. $(PY) services/voice/check.py
	PYTHONPATH=. $(PY) -m services.voice.call_model
	PYTHONPATH=. $(PY) -m services.voice.briefing
	PYTHONPATH=. $(PY) services/vision/check.py
	node web/check.mjs

bench:
	$(PY) finetune/benchmark.py report

up:
	docker compose up --build api
