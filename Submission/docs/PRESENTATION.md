# QYield — Presentation outline

## 1. The task
Learn from 300 labelled customer records, rank 100 prospects, select exactly six from the top ten, and estimate the chance of at least three subscriptions.

## 2. Prediction
Fold-local preprocessing and a fixed four-qubit ZZ feature map augment a regularized classical logistic classifier. Seed-42 OOF average precision is 0.2265, versus 0.1999 for the retained classical reference. This is development validation, not hidden-test performance or proven quantum advantage.

## 3. Selection
Ten-qubit Dicke + XY-QAOA selects C0031, C0170, C0079, C0265, C0318 and C0256. Model-estimated net value is 14.4535 units, versus 10.7082 for selecting ranks 1–6. Cost is 14/25 and capacity 16/22. Classical exact enumeration finds the same group.

## 4. Circuit resources
Structured preparation and a certified cost simplification reduce the same-topology circuit from 4259 CX gates to 426, including preparation. The result is a simulator resource comparison, not a hardware speedup.

## 5. Campaign risk
The exact independent-Bernoulli probability of at least three subscriptions is 14.1288%; 100000 noiseless quantum shots estimate 14.0470%. The combined illustrative noise model gives 16.4470%, an estimation bias rather than improved customer response.

## 6. Evidence and limits
The three result files preserve probabilities, IDs, measurements, constraints and exact benchmarks. The small training set, model-selection optimism and independence assumption limit external conclusions. Use report.html for the consolidated content and README.md for computing and reproduction instructions.
