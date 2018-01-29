DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
docker run -p 8888:8888 -v $DIR/..:/notebooks -v /Volumes/DATASETS:/mnt/DATASETS lucanode
