while getopts u:h:p:d: flag
do
    case "${flag}" in
        u) username=${OPTARG};;
        h) host=${OPTARG};;
        p) port=${OPTARG};;
        d) destintation=${OPTARG};;
    esac
done

echo "Username: $username"
echo "Host: $host"
echo "Port: $port"

echo ssh -p $port $username@$host

# cat dockercontext/setup.sh | ssh -p $port $username@$host 

scp -P $port dockercontext/requirements.txt $username@$host:$destintation/
scp -P $port dockercontext/setup.sh $username@$host:$destintation/
scp -P $port setup.ipynb $username@$host:$destintation/
scp -P $port config.yaml $username@$host:$destintation/ 
scp -P $port -r src $username@$host:$destintation/
scp -P $port -r data $username@$host:$destintation/

# kill all on that port lsof -ti:6006 | xargs kill -9
# screen -r -S "pf_tensorboard" -X quit 2>/dev/null#

#screen -r -S "tensorboard" -X quit 2>/dev/null
# Port-forwarding of tensorboard ports
# echo screen -S pf_tensorboard -dm "ssh -N -L 6006:localhost:6006  $username@$host -p $port"

# cat dockercontext/setup.sh | ssh -p $port $username@$host 

echo ssh -N -L 6006:localhost:6006  $username@$host -p $port

