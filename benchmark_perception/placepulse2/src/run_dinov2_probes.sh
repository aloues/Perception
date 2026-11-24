#!/bin/bash
# Espera a que la extraccion DINOv2 consolide (index.json) y ejecuta EXACTAMENTE los mismos probes
# que OURS/ImageNet: 3 protocolos (seed 42), 3 seeds extra en image_disjoint, control de tamano de train.
cd "$(dirname "$0")"
PY=../../../.venv/Scripts/python.exe
until [ -f ../outputs/features/dinov2_vit_b14/index.json ]; do sleep 60; done
echo "[$(date)] features DINOv2 listas" 
for proto in paper_protocol image_disjoint city_holdout; do
  [ -d ../outputs/runs/dinov2_vit_b14__${proto}__linear__seed42 ] && { echo "salto $proto (ya existe)"; continue; }
  $PY train_probe.py --encoder dinov2_vit_b14 --protocol $proto --seed 42 || echo "FALLO $proto"
done
for s in 1 2 3; do
  [ -d ../outputs/runs/dinov2_vit_b14__image_disjoint__linear__seed$s ] && continue
  $PY train_probe.py --encoder dinov2_vit_b14 --protocol image_disjoint --seed $s || echo "FALLO seed $s"
done
[ -d ../outputs/runs/ctrl__dinov2_vit_b14__paper_protocol__trainsize158312__seed42 ] || \
  $PY train_probe.py --encoder dinov2_vit_b14 --protocol paper_protocol --seed 42 --subsample-train 158312 \
     --run-id ctrl__dinov2_vit_b14__paper_protocol__trainsize158312__seed42 || echo "FALLO ctrl"
$PY collect_results.py && $PY compare_encoders_paired.py
echo "[$(date)] PROBES DINOv2 COMPLETOS"
