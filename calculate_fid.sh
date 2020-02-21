#!/bin/sh

## Requirements
# generate_images.py requires Pytorch
# fid.py requires Tensorflow

# arg 1: existing folder path of the model
# arg 2: folder path for generating images
# arg 3: size of images
# arg 4: seed
# arg 5: folder path of the calculated fid stats
# arg 6: folder path of the Inception model
# arg 7: output path for the calculated FID score
# arg 8: size of feature maps in generator

for i in `ls $1/netG_step* | sort -V`;
do 
	n=$(echo "$i" | tr -dc '0-9');
	conda activate pytorch;
	python generate_images.py --cuda --netG $i --outf "$2/$n/" --imageSize $3 --manualSeed $4 --ngf $8;
	conda deactivate;
	conda activate tensorflow;
	python fid.py "$2/$n/" "$5/fid_stats.npz" -i "$6" --prefix $n | tee -a $7;
	conda deactivate;
	rm -rf $2/$n/;
done;