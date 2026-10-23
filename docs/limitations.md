# Limitations

| # | Limitation | Impact | Status |
|---|---|---|---|
| L1 | **Customer-level leakage across the time split.** 4,549 customers have 2–3 loans (`src/ingest.py` profiling). A customer's loans can originate on different dates, so one loan can land in train and another in test, and both share identical `bur_*` (and `bur_pit_*`) customer-level features. | May slightly inflate test AUC, since the model has seen that customer's bureau profile during training. | A grouped split (by customer_id) was considered and deferred: it would shrink the test set and complicate the time-based framing the stack calls for (A2/A3 in `docs/assumptions.md`). Logged here instead; revisit if test AUC looks suspiciously strong for repeat customers specifically. |
