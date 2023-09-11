#write a script that loops the following command 10 times sfepy-run /Users/theobourdais/Desktop/Caltech/Research/model_aggregation/ModelAggregation/PDEsolvers/SfePy/helmoltz.py -o '/Users/theobourdais/Desktop/Caltech/Research/model_aggregation/ModelAggregation/PDEsolvers/SfePy/sims/acc2D_0_'+k and changes k

#for mesh in {0,'left','right','top','center'}
#mesh=0
#for i in {74..100}
#do
#    sfepy-run /Users/theobourdais/Desktop/Caltech/Research/model_aggregation/ModelAggregation/PDEsolvers/SfePy/helmoltz.py -o '/Users/theobourdais/Desktop/Caltech/Research/model_aggregation/ModelAggregation/PDEsolvers/SfePy/sims/acc2D_'$mesh"_"$i
#done

for mesh in {'top','center'}
do
    for i in {0..99}
    do
        sfepy-run /Users/theobourdais/Desktop/Caltech/Research/model_aggregation/ModelAggregation/PDEsolvers/SfePy/helmoltz.py -o '/Users/theobourdais/Desktop/Caltech/Research/model_aggregation/ModelAggregation/PDEsolvers/SfePy/sims/acc2D_'$mesh"_"$i
    done
done
