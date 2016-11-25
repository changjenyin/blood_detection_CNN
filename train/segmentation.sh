for segFile in "part"/*
do
    name="${segFile#*/}"
    name="${name%.*}"

    video=video/"$(ls video | grep "$name\.")"
    if [[ $video == *No\ such* ]]; then
        continue
    fi

    mkdir "$name"
    cnt=0
    while read line; do
        start=${line% *}
        end=${line#* }
        
        ffmpeg -ss $start -i "$video" -t "$end" "$name"/image_$cnt-%3d.jpeg < /dev/null
        cnt=$(($cnt+1))
    done < "$segFile"
done

