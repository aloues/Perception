#!/bin/bash
# EXP-061 local v2 (13/11/2026): igual que run_exp061_local.sh pero sin procesos worker (WinError 1455):
# --workers 0 --thread-workers 8. Work-dir nuevo (finetune_local_v2) con copia del checkpoint del paso 3,500;
# el original outputs/exp061/finetune_local/ queda intacto.
cd "$(dirname "$0")/.."
PY=../../.venv-gpu/Scripts/python.exe
LOG=outputs/exp061/finetune_local.log
OUT=outputs/exp061/finetune_local_v2.stdout.log
EPOCHS=${EPOCHS:-4}
export HF_HUB_OFFLINE=1 PYTHONIOENCODING=utf-8
echo "[$(date '+%F %T')] runner v2 inicia (epochs=$EPOCHS, hilos)" >> "$LOG"
for proto in ${PROTOS:-image_disjoint city_holdout}; do
  for try in 1 2 3; do
    "$PY" -u src/exp061_finetune_local.py --protocol "$proto" --epochs "$EPOCHS" --workers 0 --eval-workers 0 \
      --thread-workers 8 --work-dir outputs/exp061/finetune_local_v2 --log-file "$LOG" >> "$OUT" 2>&1 && break
    echo "[$(date '+%F %T')] $proto falló (intento $try); ver $OUT" >> "$LOG"
    sleep 30
  done
done
echo "[$(date '+%F %T')] runner v2 termina" >> "$LOG"
