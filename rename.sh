folders=$(ls ShapeNetCore.v2.PC15k/)

for folder in $folders
do
  cd /mnt/c/users/ernst/Documents/02-Work/06-Repos/PointFlow/data
  cd ./ShapNetCore.v2.PC15k/$folder/_
  mv * ../
done

