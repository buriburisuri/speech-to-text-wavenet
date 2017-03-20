# Using speech-to-text-wavenet through Docker

speech-to-text-wavenet supports `Docker` to make it easy to running through [Docker](http://www.docker.com/).

## Installing and using Docker

For general information about Docker, see [the Docker site](https://docs.docker.com/installation/).

## Download speech-to-text-wavenet docker images

Get speech-to-text-wavenet docker image

```
docker pull buriburisuri/speech-to-text-wavenet
```

## Running speech-to-text-wavenet container for shell console

Run speech-to-text-wavenet container

```
docker run -it buriburisuri/speech-to-text-wavenet 
```

## Testing speech-to-text-wavenet container

Run recognition example in the '/root/speech-to-text-wavenet'(aka '~/') directory.

```
python recognize.py --file asset/data/LibriSpeech/test-clean/1089/134686/1089-134686-0000.flac
```

# Author

Namju Kim (namju.kim@kakaocorp.com) at KakaoBrain Corp.


