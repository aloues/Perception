#!/bin/bash
# EXP-054: extrae ResNet-50 y ConvNeXt-T (congelados) y entrena EXACTAMENTE los mismos probes que RQ1
# (3 protocolos, seed 42). Espera a que EXP-048 libere la CPU.
cd "$(dirname "$0")"
PY=../../../.venv/Scripts/python.exe
until grep -q "exit=" ../outputs/exp048/models.log 2>/dev/null; do sleep 60; done
echo "[$(date)] EXP-048 terminó; empiezo CNN"
for m in resnet50 convnext_t; do
  $PY -u extract_cnn.py --model $m || { echo "FALLO extracción $m"; continue; }
done
for enc in resnet50_in1k convnext_tiny_in1k; do
  for proto in paper_protocol image_disjoint city_holdout; do
    [ -d ../outputs/runs/${enc}__${proto}__linear__seed42 ] && { echo "salto $enc $proto"; continue; }
    $PY -u train_probe.py --encoder $enc --protocol $proto --seed 42 || echo "FALLO $enc $proto"
  done
done
echo "[$(date)] EXP-054 CNN COMPLETO"
