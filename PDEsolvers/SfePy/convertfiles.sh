conv_mesh='/Users/theobourdais/miniconda3/envs/PDE/lib/python3.11/site-packages/sfepy/scripts/convert_mesh.py'
mesh_path='/Users/theobourdais/Desktop/acc_'
new_mesh_path='/Users/theobourdais/Desktop/acc2D_'

#execute /Users/theobourdais/miniconda3/envs/PDE/lib/python3.11/site-packages/sfepy/scripts/convert_mesh.py on files in list of filenames
for mesh in {0,'left','right','top','center'}
do
   python $conv_mesh -2 $mesh_path$mesh".mesh" $new_mesh_path$mesh".mesh"
done



