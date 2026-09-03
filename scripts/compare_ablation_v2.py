"""Check paired experiment controls; print event deltas and held-out metrics."""
import argparse
import json
from pathlib import Path


def compare(left, right):
    left, right = Path(left), Path(right)
    a = json.loads((left / "protocol.json").read_text())
    b = json.loads((right / "protocol.json").read_text())
    failures = []
    if {a["model"], b["model"]} != {"ingp", "vertex_v2"}:
        failures.append("Expected one ingp and one vertex_v2 run.")
    if a["protocol"] != "matched_constant" or b["protocol"] != "matched_constant":
        failures.append("Both runs must use matched_constant; original-init reference is NOT the paired control.")
    for key in ("seed", "geometry_seed", "precision", "internal_vertices_sha256",
                "exterior_vertices_sha256", "center_sha256", "scale", "source_sha256"):
        if a[key] != b[key]:
            failures.append(f"Different {key}")
    ca, cb = a["effective_config"], b["effective_config"]
    for key in (ca.keys() | cb.keys()) - {"output_path"}:
        if ca.get(key) != cb.get(key):
            failures.append(f"Different effective configuration: {key}: {ca.get(key)} vs {cb.get(key)}")
    orders = []
    for path in (left, right):
        orders.append([json.loads(line) for line in (path / "camera_order.jsonl").read_text().splitlines()])
    prefix = min(map(len, orders))
    if orders[0][:prefix] != orders[1][:prefix]:
        failures.append("Training camera order differs in the common prefix.")
    print(f"Camera-order prefix checked: {prefix} iterations; run lengths: {len(orders[0])}, {len(orders[1])}")
    for path in (left, right):
        print(f"\n{path}")
        for event in sorted((path / "diagnostics").glob("*/event.json")):
            data = json.loads(event.read_text())
            before, after = data["before"], data["after"]
            print(f"  {data['iteration']:6d} {data['event']:14s} T {before['tetrahedra']} -> {after['tetrahedra']}; "
                  f"fixed-view PSNR {before['probe_psnr']:.3f} -> {after['probe_psnr']:.3f}; "
                  f"image MAE {data['image_mean_absolute_change']:.6f}")
        result = path / "results.json"
        if result.exists():
            print("  Saved results (training PSNR and held-out metrics are distinct):", result.read_text())
    if failures:
        print("\nNOT a verified pair:")
        for failure in failures:
            print(" -", failure)
    else:
        print("\nRecorded controls match. This does NOT prove equal capacity, optimal LR, or GPU bitwise determinism.")
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ingp", required=True, type=Path)
    parser.add_argument("--vertex", required=True, type=Path)
    args = parser.parse_args()
    raise SystemExit(bool(compare(args.ingp, args.vertex)))


if __name__ == "__main__":
    main()
