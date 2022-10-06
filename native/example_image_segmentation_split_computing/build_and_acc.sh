time_info=`date +%y%m%d-%H%M`
time_info="221004-1553"
dir_name="model_"$time_info

for i in {1..10}
do
    target_dir="v1_"$dir_name"_"$i
    cd $target_dir
    for j in 0 1 2
    do
        for q in 0 2
        do
            build_cmd="python3 ../build_graph.py -c $j 0 0 0 -b 1 -q $q"
            eval $build_cmd
            json_file=$(ls | grep UNet_M'\['$j'-0-0-0]_Q\['$q']_S\[0-')
            sp=$(echo $json_file | python3 -c "cmd=input(); p=cmd.split('_S[')[-1].split(']')[0].split('-'); print(list(map(int,p))[0])")
            ep=$(echo $json_file | python3 -c "cmd=input(); p=cmd.split('_S[')[-1].split(']')[0].split('-'); print(list(map(int,p))[-1])")
            # echo "../acc_test.py -c $j 0 0 0 -q $q -p $sp $ep >> acc_result.txt"
            eval "python3 ../acc_test.py -c $j 0 0 0 -q $q -p $sp $ep >> acc_result.txt"
        done
    done
    cd ..
done

for i in {1..10}
do
    target_dir="v2_"$dir_name"_"$i
    cd $target_dir
    for j in 0 1 2
    do
        for q in 0 2
        do
            build_cmd="python3 ../build_graph.py -c $j 0 0 0 -b 1 -q $q"
            eval $build_cmd
            json_file=$(ls | grep UNet_M'\['$j'-0-0-0]_Q\['$q']_S\[0-')
            sp=$(echo $json_file | python3 -c "cmd=input(); p=cmd.split('_S[')[-1].split(']')[0].split('-'); print(list(map(int,p))[0])")
            ep=$(echo $json_file | python3 -c "cmd=input(); p=cmd.split('_S[')[-1].split(']')[0].split('-'); print(list(map(int,p))[-1])")
            # echo "../acc_test.py -c $j 0 0 0 -q $q -p $sp $ep >> acc_result.txt"
            eval "python3 ../acc_test.py -c $j 0 0 0 -q $q -p $sp $ep >> acc_result.txt"
        done
    done
    cd ..
done
