"""
Summarizes Locust *_stats.csv output from run_benchmark.sh into a markdown
table (TTFT / ITL / throughput per concurrency level), for pasting into the
portfolio README's benchmark section.

Usage:
    python summarize_results.py results/
"""
import argparse
import csv
import glob
import os


def load_stats(csv_path: str) -> dict:
    rows = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            rows[(row["Type"], row["Name"])] = row
    return rows


def summarize(results_dir: str) -> str:
    lines = [
        "| Concurrency | Requests | Failures | Avg TTFT (ms) | Avg ITL (ms) | Avg Latency (ms) | RPS |",
        "|---|---|---|---|---|---|---|",
    ]
    stats_files = sorted(glob.glob(os.path.join(results_dir, "run_*users_stats.csv")))
    if not stats_files:
        return f"No `run_*users_stats.csv` files found in {results_dir}. Run run_benchmark.sh first."

    for path in stats_files:
        users = os.path.basename(path).replace("run_", "").replace("users_stats.csv", "")
        rows = load_stats(path)

        ttft = rows.get(("TTFT", "/v1/chat/completions [stream]"))
        itl = rows.get(("ITL", "/v1/chat/completions [stream]"))
        agg = rows.get(("Aggregated", "Aggregated"))

        lines.append(
            "| {users} | {reqs} | {fails} | {ttft} | {itl} | {lat} | {rps} |".format(
                users=users,
                reqs=agg["Request Count"] if agg else "-",
                fails=agg["Failure Count"] if agg else "-",
                ttft=ttft["Average Response Time"] if ttft else "-",
                itl=itl["Average Response Time"] if itl else "-",
                lat=agg["Average Response Time"] if agg else "-",
                rps=agg["Requests/s"] if agg else "-",
            )
        )
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", nargs="?", default="results")
    args = parser.parse_args()
    print(summarize(args.results_dir))
