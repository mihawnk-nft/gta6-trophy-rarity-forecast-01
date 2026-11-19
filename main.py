#!/usr/bin/env python3
"""Forecast GTA VI trophy earn rates on Exophase one week after launch.

Reads every ``*.in`` case in $MAKEFUN_INPUT_DIR (or /input) and writes one
``<id>.out`` forecast per case to $MAKEFUN_OUTPUT_DIR (or /output).

Method (summary): an Exophase earn rate is a ratio, earned / tracked, and
the denominator jumps whenever Exophase refreshes its counts, so the
numerator (players who earned each trophy) and the denominator (players
tracked) are forecast separately in log-log time, then divided.  The
data-driven rate is blended in logit space with a per-class day-7 prior
anchor (story chapters complete far faster than grinds), and the spread
is wide on purpose: launch-week refreshes move individual trophies by
roughly 0.3-0.9 logit, and a narrow band is what loses on this scoring
rule.  Standard library only, no network, fully deterministic.
"""

import json
import math
import os
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP

# Standard-normal quantiles for the five scored levels.
Z_SCORES = (-1.645, -0.674, 0.0, 0.674, 1.645)

# Spread pieces, in logit units (see the design note in the README).
REFRESH_SCALE = 0.7   # how far one Exophase refresh moves a trophy
PRIOR_SCALE = 0.5     # uncertainty of the class prior anchor
SIGMA_FLOOR = 0.8
SIGMA_CAP = 2.5

# Weight of the class-mean slope when shrinking a trophy's own slope.
CLASS_SLOPE_WEIGHT = 10.0

# A fit counts as "tight" for the high data weight when its residual
# spread stays under this many logit units.
TIGHT_FIT = 0.25


def parse_time(text):
    """Parse an ISO 8601 UTC timestamp into an aware datetime."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def hours_since(origin, moment):
    return (moment - origin).total_seconds() / 3600.0


def logit(rate_pct):
    """logit of a percent in (0, 100); the caller clamps first."""
    p = rate_pct / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(x):
    """Inverse logit, returned as a percent in (0, 100)."""
    if x >= 0.0:
        return 100.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return 100.0 * e / (1.0 + e)


def clamp(value, low, high):
    return low if value < low else high if value > high else value


def round4(value):
    """Round to 4 decimals, half-up, as a plain float."""
    return float(Decimal(repr(value)).quantize(
        Decimal("0.0001"), rounding=ROUND_HALF_UP))


def least_squares(xs, ys):
    """Ordinary least squares of y on x.

    Returns (slope, intercept, residual_std).  With fewer than two
    distinct x values the slope is 0 and the intercept is the mean of y.
    """
    n = len(xs)
    if n == 0:
        return 0.0, 0.0, 0.0
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    if sxx <= 0.0:
        resid = math.sqrt(sum((y - mean_y) ** 2 for y in ys) / n) if n else 0.0
        return 0.0, mean_y, resid
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = mean_y - slope * mean_x
    if n > 2:
        resid = math.sqrt(
            sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
            / (n - 2)
        )
    else:
        resid = 0.0
    return slope, intercept, resid


def classify(trophy):
    """Return (day-7 prior anchor percent, class label) for one trophy.

    Classification is by description keywords plus tier, grounded in
    long-run GTA V / RDR2 rates: story chapters are finished by most
    story players inside a week, strand finals need many hours, grinds
    and collectibles need dedicated effort, the platinum needs 200+
    hours, and the opening mission is done by almost everyone.
    """
    tier = trophy.get("tier", "")
    desc = trophy.get("description", "").lower()

    if tier == "platinum":
        return 0.1, "platinum"
    if "gone fishing" in desc:
        return 90.0, "opening"
    if "chapter 1" in desc:
        return 60.0, "story"
    if "chapter 2" in desc:
        return 45.0, "story"
    if "chapter 3" in desc:
        return 35.0, "story"
    if "chapter 4" in desc:
        return 25.0, "story"
    if "100% completion" in desc:
        return 0.5, "completion"
    if "complete the final mission" in desc and "for " not in desc:
        return 12.0, "final"

    # Grinds and collectibles first: they also start with "Complete".
    if "250 stunts" in desc:
        return 2.0, "grind"
    if "gold medal in all shooting range" in desc:
        return 4.0, "grind"
    if "8 mission strands" in desc:
        return 8.0, "grind"
    if "3 collectible card sets" in desc:
        return 2.0, "grind"
    if "100 locations" in desc:
        return 2.0, "grind"
    if "5 bank vaults" in desc:
        return 5.0, "grind"
    if "50 challenges" in desc:
        return 2.0, "grind"

    # Easy one-off bronze misc.
    if "try each type of activity" in desc:
        return 30.0, "easy"
    if "arcade game" in desc:
        return 20.0, "easy"
    if "meditate" in desc:
        return 8.0, "easy"

    # Activities: races, fishing, hunting, exploring, money, robbery.
    if "top fun drone race" in desc or "win 20 different races" in desc:
        return 5.0, "activity"
    if "beat cal" in desc:
        return 12.0, "activity"
    if "hunt a unique animal" in desc:
        return 8.0, "activity"
    if "explore all of leonida" in desc:
        return 8.0, "activity"
    if "discover 100 vehicles" in desc:
        return 5.0, "activity"
    if "laundering marked bills" in desc:
        return 4.0, "activity"
    if "spend $100,000" in desc:
        return 12.0, "activity"
    if "purchase all garage properties" in desc:
        return 4.0, "activity"
    if "register 10 vehicles" in desc:
        return 8.0, "activity"
    if "rob 3 shops" in desc:
        return 15.0, "activity"
    if "rob a jewelry store" in desc:
        return 8.0, "activity"
    if "rob an armored truck" in desc:
        return 8.0, "activity"
    if "rob an illegal operation" in desc:
        return 8.0, "activity"
    if "4 star wanted level" in desc:
        return 10.0, "activity"
    if "combat focus" in desc:
        return 8.0, "activity"
    if "unscheduled departure" in desc:
        return 8.0, "activity"
    if "rhinos" in desc:
        return 5.0, "activity"

    # Remaining one-off bronze misc (cocktail, showroom, bowling, ...).
    if any(word in desc for word in (
        "cocktail", "showroom", "bowling", "partnership with mass appeal",
        "choices made", "second time's the charm",
        "what the hell just happened", "melee drive-by",
    )):
        return 12.0, "easy"

    # Any other named-mission / strand-final completion.
    if desc.startswith("complete "):
        return 8.0, "mission"

    return 8.0, "activity"


def forecast_case(case):
    """Forecast one case; returns the {"trophies": [...]} object."""
    trophies = case["trophies"]
    history = case.get("history", [])
    horizon_days = case["horizon_minutes"] / 1440.0
    target = parse_time(case["target"])

    release = case.get("game", {}).get("release_date", "2026-11-19")
    origin = datetime.fromisoformat(release + "T00:00:00+00:00")
    target_h = hours_since(origin, target)
    target_x = math.log1p(target_h)

    # Window: reads from the last 24 hours of history (all of them when
    # fewer than 3 reads exist).
    if history:
        last_at = max(parse_time(read["at"]) for read in history)
        window = [
            read for read in history
            if (last_at - parse_time(read["at"])).total_seconds() <= 24 * 3600
        ]
        if len(window) < 3:
            window = list(history)
    else:
        window = []

    xs = [math.log1p(hours_since(origin, parse_time(read["at"])))
          for read in window]

    # --- Denominator: players tracked, log N on log(1 + hours). ---
    n_hat = None
    if window:
        n_ys = [math.log(max(read["players_tracked"], 1)) for read in window]
        n_slope, n_intercept, _ = least_squares(xs, n_ys)
        n_hat = math.exp(n_intercept + n_slope * target_x)
        last_n = max(max(read["players_tracked"] for read in window), 1)
        # The tracked population only grows, and a log-log line fit on a
        # steep launch-night slope must not extrapolate to an absurd
        # denominator (which would crush every data-driven rate toward
        # zero): floor at the last observed count, cap at 50x it.
        n_hat = clamp(n_hat, last_n, 50.0 * last_n)

    # --- Numerator: earned counts per trophy, log(E + 0.5) on time. ---
    # First pass: each trophy's own fit, so class slopes can be averaged.
    own_fits = {}
    for index, trophy in enumerate(trophies):
        points = []
        for read, x in zip(window, xs):
            rate = read["rates"][index]
            if rate is None:
                continue
            earned = rate * max(read["players_tracked"], 1) / 100.0
            points.append((x, math.log(earned + 0.5), earned))
        if not points:
            own_fits[index] = None
            continue
        px = [p[0] for p in points]
        py = [p[1] for p in points]
        slope, intercept, resid = least_squares(px, py)
        own_fits[index] = {
            "slope": slope,
            "intercept": intercept,
            "resid": resid,
            "earned_first": points[0][2],
            "earned_last": points[-1][2],
            "earned_total": sum(p[2] for p in points),
        }

    # Class-mean slopes, for shrinking tiny-count trophies toward their
    # class (a trophy earned by 4 players says little on its own).
    class_slopes = {}
    class_counts = {}
    for index, trophy in enumerate(trophies):
        fit = own_fits[index]
        if fit is None:
            continue
        _, label = classify(trophy)
        class_slopes[label] = class_slopes.get(label, 0.0) + fit["slope"]
        class_counts[label] = class_counts.get(label, 0) + 1
    for label in class_slopes:
        class_slopes[label] /= class_counts[label]

    rows = []
    for index, trophy in enumerate(trophies):
        anchor, label = classify(trophy)
        fit = own_fits[index]

        data_rate = None
        fit_spread = 0.5
        data_weight = 0.0
        if fit is not None and n_hat:
            # Shrink the trophy's slope toward its class mean, weighted
            # by how many players actually earned it.
            class_slope = class_slopes.get(label, fit["slope"])
            weight = fit["earned_total"]
            slope = (
                weight * fit["slope"] + CLASS_SLOPE_WEIGHT * class_slope
            ) / (weight + CLASS_SLOPE_WEIGHT)
            earned_hat = math.exp(fit["intercept"] + slope * target_x) - 0.5
            earned_hat = max(earned_hat, 0.0)
            if fit["earned_total"] > 0.0:
                data_rate = clamp(
                    100.0 * earned_hat / n_hat, 0.01, 99.99)
            fit_spread = clamp(fit["resid"], 0.1, 1.0)
            # Data weight from how much the trophy actually moved.
            growth = (fit["earned_last"] + 0.5) / (fit["earned_first"] + 0.5)
            if growth >= 3.0 and fit["resid"] < TIGHT_FIT:
                data_weight = 0.7
            elif growth >= 1.5:
                data_weight = 0.4
            else:
                data_weight = 0.2

        # Blend data and prior in logit space.
        if data_rate is None:
            median_logit = logit(anchor)
        else:
            median_logit = (
                data_weight * logit(data_rate)
                + (1.0 - data_weight) * logit(anchor)
            )

        # Spread: random-walk growth with horizon, fit residual,
        # refresh-move scale and prior uncertainty, combined in
        # quadrature; floored so a flat history never yields a
        # house-narrow band.
        sigma = math.sqrt(
            (0.5 * math.sqrt(horizon_days)) ** 2
            + fit_spread ** 2
            + REFRESH_SCALE ** 2
            + PRIOR_SCALE ** 2
        )
        sigma = clamp(sigma, SIGMA_FLOOR, SIGMA_CAP)

        quantiles = []
        previous = 0.01
        for z in Z_SCORES:
            q = clamp(inv_logit(median_logit + z * sigma), 0.01, 99.99)
            q = round4(q)
            # Rounding must never break the non-decreasing order.
            if q < previous:
                q = previous
            quantiles.append(q)
            previous = q
        rows.append({"id": trophy["id"], "q": quantiles})

    rows.sort(key=lambda row: row["id"])
    return {"trophies": rows}


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(output_dir, exist_ok=True)
    names = sorted(
        name for name in os.listdir(input_dir) if name.endswith(".in")
    )
    for name in names:
        with open(os.path.join(input_dir, name), "r", encoding="utf-8") as fh:
            case = json.load(fh)
        forecast = forecast_case(case)
        out_name = name[:-3] + ".out"
        with open(os.path.join(output_dir, out_name), "w",
                  encoding="utf-8") as fh:
            fh.write(json.dumps(forecast, separators=(",", ":")))


if __name__ == "__main__":
    main()
