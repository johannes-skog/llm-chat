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

scp -P $port dockercontext/requirements.txt $username@$host:$destintation/
scp -P $port dockercontext/setup.sh $username@$host:$destintation/
scp -P $port setup.ipynb $username@$host:$destintation/
scp -P $port config.yaml $username@$host:$destintation/ 
scp -P $port -r src $username@$host:$destintation/
scp -P $port -r data $username@$host:$destintation/


# Port-forwarding of tensorboard ports
ssh -fNT -L 6006:localhost:6006  $username@$host -p $port