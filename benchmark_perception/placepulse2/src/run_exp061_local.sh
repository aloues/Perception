#!/bin/bash
# EXP-061 local: fine-tuning DINOv2 K=2 en la GPU del laptop, image_disjoint y luego city_holdout.
# Reanudable: re-ejecutar continúa desde outputs/exp061/finetune_local/<run>/last.pt; salta protocolos con metrics.json.
# Si el proceso cae (p. ej. error CUDA), reintenta hasta 3 veces desde el último checkpoint.
# Uso: bash src/run_exp061_local.sh            (EPOCHS=2 MAX_TRAIN_PAIRS=80000 bash ... para cambiar la config)
cd "$(dirname "$0")/.."
PY=../../.venv-gpu/Scripts/python.exe
LOG=outputs/exp061/finetune_local.log
OUT=outputs/exp061/finetune_local.stdout.log      # stdout/stderr crudo (tracebacks)
EPOCHS=${EPOCHS:-4}
MAXP=${MAX_TRAIN_PAIRS:+--max-train-pairs $MAX_TRAIN_PAIRS}
export HF_HUB_OFFLINE=1 PYTHONIOENCODING=utf-8    # pesos timm ya en caché; sin red
mkdir -p outputs/exp061
echo "[$(date '+%F %T')] runner inicia (epochs=$EPOCHS $MAXP)" >> "$LOG"
for proto in image_disjoint city_holdout; do
  for try in 1 2 3; do
    "$PY" -u src/exp061_finetune_local.py --protocol "$proto" --epochs "$EPOCHS" $MAXP --log-file "$LOG" >> "$OUT" 2>&1 && break
    echo "[$(date '+%F %T')] $proto falló (intento $try); ver $OUT" >> "$LOG"
    sleep 30
  done
done
echo "[$(date '+%F %T')] runner termina" >> "$LOG"
