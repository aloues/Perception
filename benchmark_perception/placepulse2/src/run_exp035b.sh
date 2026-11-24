#!/bin/bash
# EXP-035B: construye las features multicapa, entrena el probe fijo de RQ1 en 3 protocolos y compara con DINOv2 CLS.
cd "$(dirname "$0")"
PY=../../../.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8
$PY -u exp035b_multilayer.py --stage build || exit 1
for enc in dinov2_cls_avg dinov2_cls_avg_L9_11; do
  for proto in image_disjoint city_holdout paper_protocol; do
    [ -d ../outputs/runs/${enc}__${proto}__linear__seed42 ] && continue
    $PY -u train_probe.py --encoder $enc --protocol $proto --seed 42 | grep -E "test|Guardado" || echo "FALLO $enc $proto"
  done
done
$PY -u exp035b_multilayer.py --stage compare
echo "EXP-035B COMPLETO"
