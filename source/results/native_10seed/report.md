# Native ns-3.48 RU Scheduler Results

## Experiment

- 10 paired unseen RNG seeds per scenario and policy, 360 network runs total.
- 2 s warm-up and 1 s measurement per run.
- HE 20 MHz, one NSS, one best-effort TID, 1,000-byte UDP packets.
- Default RR uses its default four-station scheduling limit.
- Oracle and learned RU policies permit up to nine stations and arbitrary legal
  20 MHz mixed-RU partitions.
- Training used different seeds, packet sizes, loads, and MCS/topology points.
- Intervals below are paired two-sided 95% Student-t confidence intervals.

## Throughput model versus default RR

| Scenario | RR Mbps | ML-T Mbps | Paired gain Mbps | RR mean delay ms | ML-T mean delay ms | Paired delay change ms |
|---|---:|---:|---:|---:|---:|---:|
| Sparse | 6.31 | 6.31 | 0.00 +/- 0.00 | 0.56 | 0.87 | +0.31 +/- 0.00 |
| Normal | 49.83 | 49.48 | -0.35 +/- 0.30 | 10.91 | 3.70 | -7.21 +/- 1.77 |
| Saturated | 63.99 | 92.12 | +28.13 +/- 0.82 | 562.47 | 518.26 | -44.21 +/- 11.67 |
| Low MCS | 17.35 | 23.04 | +5.69 +/- 0.22 | 1007.83 | 840.41 | -167.42 +/- 48.53 |
| 4 STAs | 69.51 | 88.79 | +19.28 +/- 0.03 | 419.34 | 7.16 | -412.18 +/- 0.42 |
| 7 STAs | 43.02 | 61.59 | +18.57 +/- 0.78 | 782.48 | 356.67 | -425.81 +/- 23.76 |

ML-T is within 0.02 Mbps of Oracle-T in every scenario. At normal load all
policies are offer-limited; the small apparent RR throughput excess includes
warm-up packets delivered inside the measurement window and is not evidence of
higher capacity.

## Delay-aware model

ML-D improves throughput and mean delay over RR in all overloaded scenarios,
but does not uniformly imitate Oracle-D. It loses 9.74 Mbps to Oracle-D with
four STAs and 8.54 Mbps with seven STAs. Under saturation it delivers 83.75 Mbps
with Jain fairness 0.926, compared with 92.12 Mbps and fairness 0.907 for ML-T.
The result is a throughput/fairness/tail-pressure tradeoff, not uniform
superiority of the chosen age weight.

## Serial host latency

An independent 18-run serial campaign avoids contention from parallel network
evaluation:

| Component | Mean us | Mean run-level p99 us |
|---|---:|---:|
| Queue state and features | 163.7 | 195.4 |
| Model inference | 175.3 | 192.8 |
| Legal projection | 483.0 | 640.9 |
| Complete decision | 824.5 | 1016.0 |

The mean run-level maximum is 1348.9 us and the global observed maximum is
3261.6 us. At a declared 1500 us budget, the mean deadline-miss rate is 0.12%.
Approximately five empty-action fallbacks per
run occur during association/BlockAck setup; the legal decoder does not produce
invalid nonempty actions.

These wall-clock measurements support feasibility on this host and software
build only. They are not a Jetson Nano or production-AP hard-realtime guarantee.

## Interpretation

The corrected implementation supports the research hypothesis in overloaded,
controlled ns-3 conditions: offline optimization can label native AP states,
and a compact projected set network can preserve the throughput oracle's action
utility with substantially lower and less variable decision cost.
It does not establish universal superiority, direct average-delay optimality,
or robustness to mobility, heterogeneous time-varying MCS, frequency-selective
interference, multiple TIDs/NSS values, or competing BSSs.
