(cd src && make)
bin/main | tee out/rate_swap.json
python3 script/process.py out/rate_swap.json fig/
for file in fig/*.pdf; do xdg-open $file; done
