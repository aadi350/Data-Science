# Zero Padding With Awk

```bash
~/projects/awk_pad ❯ cat out.csv             
a,Y,1
b,N,10
c,Y,12223253
```

```bash
~/projects/awk_pad ❯ cat out_clean.csv
a,Y,00000000001
b,N,00000000010
c,Y,00012223253

```bash
~/p/awk_pad ❯ awk -F ',' '{OFS=","};{print $1, $2, sprintf("%011d", $3)}' out.csv > out_clean.csv
```
