# First Witness — stubs until services land.
.PHONY: up demo reset bench api check ambient help

help:
	@echo "make api     # run services/api on :8080"
	@echo "make check   # brain + api + voice + osnet self-checks"
	@echo "make ambient # optional placeholder clips, not the three-camera demo"
	@echo "make up      # docker compose up api"
	@echo "make demo    # pre-warm + scenario (partial)"
	@echo "make reset   # reset demo state (not wired yet)"
	@echo "make bench   # run benchmarks (not wired yet)"

api:
	PYTHONPATH=. python3 -m services.api --host 127.0.0.1 --port 8080

check:
	python3 services/brain/check.py
	python3 services/api/check.py
	python3 services/voice/check.py
	python3 services/vision/osnet.py

ambient:
	python3 scripts/make_ambient_placeholders.py

up:
	docker compose up --build api

demo:
	@echo "Start api (make api), then: python3 -m http.server 8090 --bind 127.0.0.1 --directory web"
	@echo "Open http://127.0.0.1:8090/?ws=ws://127.0.0.1:8080/ws"

reset:
	@echo "TODO: POST reset via WS cmd — use dashboard Reset or send {\"cmd\":\"reset\"}"

bench:
	@echo "TODO: make bench — Naman"
	@exit 1
