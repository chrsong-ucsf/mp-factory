import re
with open('results/scaling_sweep/SCALING_SWEEP_REPORT.md', 'r') as f:
    text = f.read()

table_new = """| Cohort Size (N) | Best Mean Dice (20-case Consensus) | Best Mean Dice (JHU Gold Standard) |
|-----------------|------------------------------------|------------------------------------|
| 4               | ~0.027                             | ~0.023                             |
| 10              | ~0.017                             | ~0.016                             |
| 15              | ~0.017                             | ~0.020                             |
| 20              | ~0.025                             | ~0.022                             |
| 30              | ~0.049                             | ~0.039                             |
| 50              | ~0.040                             | ~0.033                             |"""

text = re.sub(r'\| Cohort Size \(N\) \|.*?\| 50              \| TBD                                \| TBD                                \|', table_new, text, flags=re.DOTALL)

text = text.replace('*Pending JHU Gold Standard Evaluation.*', 'The JHU Gold Standard Evaluation shows an extremely low performance ceiling similarly impacted by the lack of small bowel annotations in the underlying training data. The discrepancy between Consensus evaluation and Radiologist Evaluation remains minimal given the baseline floor effect.')

with open('results/scaling_sweep/SCALING_SWEEP_REPORT.md', 'w') as f:
    f.write(text)
