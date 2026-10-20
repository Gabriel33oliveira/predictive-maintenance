# Predictive Maintenance

Simulated pump and motor sensor streams with anomaly detection, RUL estimates and a ranked maintenance work-order list.

Predictive Maintenance is a self-contained Python toolkit that generates synthetic vibration and temperature data for rotating equipment such as pumps and motors. It can inject realistic degradation patterns into that data: gradual bearing wear, sudden faults and overheating. Rolling-statistics anomaly detection flags abnormal readings, and a linear trend fit projects each machine's health indicator toward a failure threshold to estimate its remaining useful life (RUL). Machines are then scored and ranked into a prioritized maintenance work-order list. You drive everything from a command-line interface, and the results come out as CSV/JSON files plus a standalone HTML report with charts. The whole thing runs offline on synthetic data with no external services, so you can reproduce any run from a single random seed.
