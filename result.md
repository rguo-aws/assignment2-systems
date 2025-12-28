DPP benchmark 

stepup:
backend : NCCL
RTX 3090 * 2
num_steps = 5, warm up = 5

Naive DPP : 
total time per step : 41.91ms
comm time per step : 3.76ms
time spent on comm : 8.96%

flat DPP
total time per step : 37.53ms
comm time per step : 0.31ms
time spent on comm : 0.83%
