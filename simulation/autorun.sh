#!/bin/bash  
  
while IFS= read -r line; do  
  if [[ -n "$line" ]]; then  
    python simulation/bdd100k-fl-sim.py -c "$line" -n 15
    python simulation/bdd100k-fl-eval.py -c "$line"
  fi  
done < "simulation/waiting_model_queue.txt"