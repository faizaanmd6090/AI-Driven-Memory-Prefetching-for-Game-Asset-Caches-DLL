"""
Generate a synthetic asset access trace for testing prefetchers.
Each row: time_ms, asset_id, size_bytes
"""
import argparse, csv, random

def gen_trace(n_events=1000, n_assets=300, n_sectors=20, base_size=64*1024):
    """
    Generate a fake access pattern:
      - assets grouped into sectors (regions)
      - player moves between sectors
      - each sector favors a subset of assets
    """
    sector_assets = []
    for s in range(n_sectors):
        pop = random.sample(range(n_assets), k=min(2000, n_assets))
        sector_assets.append(pop)

    t = 0
    cur_sector = 0
    for i in range(n_events):
        # occasionally switch sector (simulate player moving)
        if random.random() < 0.02:
            cur_sector = random.randrange(n_sectors)
            t += random.randint(5, 25)
        else:
            t += random.randint(1, 3)

        # pick asset: mostly from this sector
        if random.random() < 0.6:
            aid = random.choice(sector_assets[cur_sector])
        else:
            aid = random.randrange(n_assets)

        size = int(base_size * random.choice([1, 1, 2, 4]))
        yield {"t_ms": t, "asset_id": f"a{aid}", "size_bytes": size, "sector": cur_sector}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output CSV path")
    ap.add_argument("--events", type=int, default=1000)
    ap.add_argument("--assets", type=int, default=300)
    ap.add_argument("--sectors", type=int, default=20)
    ap.add_argument("--base_size", type=int, default=64*1024)
    args = ap.parse_args()

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["t_ms", "asset_id", "size_bytes", "sector"])
        w.writeheader()
        for row in gen_trace(args.events, args.assets, args.sectors, args.base_size):
            w.writerow(row)
    print(f"Wrote synthetic trace to {args.out}")

if __name__ == "__main__":
    main()