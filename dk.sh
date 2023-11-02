#!/bin/bash

image="nvcr23.08_v2"
container="cnn_resnet_0"
image_location=$2
gpus=0

if [ -z $container ] ; then
    container="unknown"
fi

if [ -z $image_location ] ; then
    image_location="."
fi

echo "==================================="
echo "option 1: create docker image"
echo "option 2: create docker container and access recent one"
echo "option 3: only access"
echo "option 4: kill docker container"
echo "option 5: modify image/container etc"
echo "else: out"
echo "image = $image"
echo "image location = $image_location"
echo "container = $container"
echo "==================================="
read option

if [ $option == 5 ] ; then
    n=0
else
    n=3
fi

while [ $n != 3 ]
do
if [ $option == 5 ] ; then 
    echo "modify image(1) | container(2) | exit(3)"
    read nn
    if [ $nn == 1 ] ; then
        read -p "new image name: " ni
        image=$ni
    elif [ $nn == 2 ] ; then
        read -p "new container name: " nc
        container=$nc
    elif [ $nn == 3 ] ; then
        echo "image = $image"
        echo "container = $container"
    else
        echo "$nn no option"
    fi
    n=$nn
fi
done

if [ $option == 5 ] ; then
    echo "==================================="
    echo "option 1: create docker image"
    echo "option 2: create docker container and access recent one"
    echo "option 3: only access"
    echo "option 4: kill docker container"
    echo "option 5: modify image/container etc"
    echo "else: out"
    echo "image = $image"
    echo "image location = $image_location"
    echo "container = $container"
    echo "==================================="
    read option
fi

# docker image creation
if [ $option == 1 ] ; then
    cat ./dockerfile
    docker build -t $image --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) $image_location
# docker container creation and access maken one
elif [ $option == 2 ] ; then 
    # volume(-v) means make container work directory to $pwd
    # mount means the data location, unless the image dataset always downloaded 
    #mkdir $container
    docker run -itd --name $container --mount type=bind,source=/home/jsw/data,target=/home/jsw/data,readonly --rm --ipc=host --gpus "device=$gpus" --workdir /app -v "$(pwd)":/app --user $(id -u):$(id -g) $image /bin/bash
    docker exec -it $container /bin/bash
elif [ $option == 3 ] ; then
    docker exec -it $container /bin/bash
# docker kill container
elif [ $option == 4 ] ; then
    docker kill $container
    echo "docker container ($container) has been removed"
else
    exit
fi

