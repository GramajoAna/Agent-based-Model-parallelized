for((i=100;i<205;i++))
do 
	echo $i; 
	mkdir run$i; 
	cp mosquitas jobGPUtest run$i; 
	cd run$i; 
	qsub jobGPUtest $i; 
	cd ../ ;
done
