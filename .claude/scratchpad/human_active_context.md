## Technical Challenge - Detection

The hardest part of this project is getting the system to to detect the paper and what's written on the paper in real time.

- we tried using openCV and template matching but it struggled to find the paper.
- Paper detection may struggle in poor lighting conditions.
- the llm based vision detection was able to find the paper and what was written but it was too slow for real time gameplay.
- currently trying: using moondream (see docs at https://docs.moondream.ai/)
