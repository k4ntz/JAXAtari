total=`seq 0 $(expr $1 - 1)`
shift
for i in $total
do
  numactl --cpunodebind=$i --membind=$i -- $@ &
done
wait