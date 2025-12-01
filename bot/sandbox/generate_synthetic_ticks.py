import csv
import random
import time
from pathlib import Path


def generate_ticks(symbol="BTCUSDT", n=5000, start_price=43000):
    out_dir = Path("data/ticks")
    out_dir.mkdir(parents=True, exist_ok=True)

    file_path = out_dir / f"{symbol}_synthetic.csv"

    with open(file_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "price", "qty", "side"])

        price = start_price

        ts = int(time.time() * 1000)

        for _ in range(n):
            # synthetic micro price movement
            price += random.uniform(-1, 1)

            qty = random.uniform(0.001, 0.03)
            side = random.choice(["buy", "sell"])

            writer.writerow([ts, round(price, 2), qty, side])

            ts += random.randint(50, 300)  # random tick spacing

    print(f"[OK] Generated {n} synthetic ticks at {file_path}")


if __name__ == "__main__":
    generate_ticks()
