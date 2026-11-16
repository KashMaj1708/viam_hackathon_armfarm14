"""Per-run metrics for track3_sort: timings per item, run totals, and where inference runs.

Writes, per run (runs/<timestamp>/):
  judge_log.csv   one line per item: number, decision, confidence, timestamp (+ first motion, latency)
  items.csv       per-attempt timings and outcome
  summary.json    run totals; also printed at the end of the run
"""
import csv
import json
import os
import statistics
import time
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
RUN_LIMIT_S = 300                      # judged run is five minutes


def _ms(a, b):
    return None if a is None or b is None else round((b - a) * 1000)


class Run:
    def __init__(self, dry=False):
        self.t0 = time.monotonic()
        self.started = datetime.now()
        self.dir = os.path.join(HERE, "runs", "dry_runs" if dry else "", self.started.strftime("%Y%m%d_%H%M%S"))
        os.makedirs(self.dir, exist_ok=True)
        self.items = []
        self.infer_ms = []             # every anomaly-model call (client round trip)
        self.detect_s = []
        self.device = {}
        self.judge = open(os.path.join(self.dir, "judge_log.csv"), "w", newline="")
        self.jw = csv.writer(self.judge)
        self.jw.writerow(["number", "decision", "confidence", "timestamp", "first_motion", "latency_ms",
                          "anomaly_score", "decided_by"])
        self._warned = False

    def elapsed(self):
        return time.monotonic() - self.t0

    def clock(self):
        e = self.elapsed()
        if e > RUN_LIMIT_S - 30 and not self._warned:
            self._warned = True
            print(f"*** {e:.0f}s elapsed: 30 s left in a five-minute judged run ***")
        return f"[{int(e // 60)}:{int(e % 60):02d}]"

    async def probe_device(self, machine, mlmodel="ov-anomaly", diag="ov-diag"):
        """Where does the anomaly model run? Asks the OpenVINO diagnostics service for the
        compiled execution device (no extra client packages needed)."""
        try:
            from viam.services.generic import Generic
            try:
                g = Generic.from_robot(machine, diag)
            except Exception:                     # modular service not in the client's resource map
                from viam.services.generic.client import GenericClient
                g = GenericClient(diag, machine._channel)
            self._diag = g
        except Exception as e:
            self.device["anomaly_model"] = f"unknown ({e})"
            return
        try:
            devs = (await g.do_command({"command": "list_devices"})).get("devices", [])
            self.device["devices"] = {d["name"]: d["full_name"] for d in devs}
        except Exception as e:
            self.device["devices"] = f"unknown ({e})"
        try:
            p = await g.do_command({"command": "get_compiled_properties", "mlmodel_name": mlmodel})
            ex = p.get("execution_devices") or p.get("EXECUTION_DEVICES") or p.get("compile_device") or p.get("device")
            self.device["anomaly_model"] = f"OpenVINO, execution devices {ex}"
        except Exception as e:
            self.device["anomaly_model"] = f"unknown ({e})"
        self.device["agent_review"] = "Claude Opus 5.5 via Anthropic API (cloud), overlapped with the pick"
        self.device["detection"] = "RealSense depth clustering (viam-server, CPU) + OpenCV colour segmentation (laptop CPU)"
        gpu = self.device["devices"].get("GPU", "") if isinstance(self.device.get("devices"), dict) else ""
        print(f"Inference: {self.device['anomaly_model']} | GPU: {gpu}")

    def judge_row(self, number, decision, conf, t_dec, t_motion, lat_ms, score, by):
        self.jw.writerow([number, decision, f"{conf:.3f}", t_dec, t_motion, lat_ms, f"{score:.3f}", by])
        self.judge.flush()

    def add(self, rec):
        rec["t_run_s"] = round(self.elapsed(), 1)
        self.items.append(rec)

    async def finish(self):
        server_stats = None
        try:
            server_stats = await self._diag.do_command({"command": "stats", "mlmodel_name": "ov-anomaly"})
        except Exception:
            pass
        total = self.elapsed()
        done = [r for r in self.items if r.get("outcome") == "sorted"]
        lat = [r["decision_to_motion_ms"] for r in self.items if r.get("decision_to_motion_ms") is not None]
        cyc = [r["cycle_s"] for r in done if r.get("cycle_s")]
        rv = [r["review_s"] for r in self.items if r.get("review_s") is not None]
        notes = [r.get("review_note") for r in self.items if r.get("review_note")]
        s = {
            "run_started": self.started.isoformat(timespec="seconds"),
            "total_time_s": round(total, 1),
            "parts_sorted": len(done),
            "seconds_per_part": round(total / len(done), 1) if done else None,
            "mean_cycle_s": round(statistics.mean(cyc), 1) if cyc else None,
            "decisions": {k: sum(1 for r in done if r.get("decision") == k) for k in ("good", "defective")},
            "grasp_attempts": len(self.items),
            "missed_grasps": sum(1 for r in self.items if r.get("outcome") == "missed"),
            "decision_to_first_motion_ms": {"mean": round(statistics.mean(lat)) if lat else None,
                                            "max": max(lat) if lat else None},
            "anomaly_inference_ms_client": {"mean": round(statistics.mean(self.infer_ms)) if self.infer_ms else None,
                                            "p90": round(sorted(self.infer_ms)[int(0.9 * (len(self.infer_ms) - 1))])
                                            if self.infer_ms else None, "calls": len(self.infer_ms)},
            "anomaly_inference_ms_on_device": server_stats,
            "mean_detect_s": round(statistics.mean(self.detect_s), 1) if self.detect_s else None,
            "agent_review": {"calls": len(rv), "mean_s": round(statistics.mean(rv), 1) if rv else None,
                             "agrees": notes.count("agent_agrees"), "overrides": notes.count("agent_override"),
                             "unsure": notes.count("agent_unsure"), "unavailable": notes.count("model")},
            "within_5_min": total <= RUN_LIMIT_S,
            "compute": self.device,
        }
        with open(os.path.join(self.dir, "summary.json"), "w") as f:
            json.dump(s, f, indent=2)
        if self.items:
            keys = sorted({k for r in self.items for k in r})
            with open(os.path.join(self.dir, "items.csv"), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                w.writerows(self.items)
        self.judge.close()
        print("\n==== RUN SUMMARY ====")
        print(f"Total {s['total_time_s']} s | sorted {s['parts_sorted']} | {s['seconds_per_part']} s/part "
              f"| good {s['decisions']['good']} / defective {s['decisions']['defective']} "
              f"| missed grasps {s['missed_grasps']}")
        print(f"Decision -> first motion: mean {s['decision_to_first_motion_ms']['mean']} ms, "
              f"max {s['decision_to_first_motion_ms']['max']} ms")
        print(f"Anomaly inference: {s['anomaly_inference_ms_client']['mean']} ms/frame (client round trip); "
              f"on device: {(server_stats or {}).get('mean_ms', 'n/a') if isinstance(server_stats, dict) else 'n/a'} ms")
        print(f"Agent review: {s['agent_review']}")
        print(f"Within 5 min: {s['within_5_min']}  |  files: {self.dir}")
        return s
