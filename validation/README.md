# Release validation

The ARM64 image builds the optional C++ NMI backend from the release source and
runs the complete core suite from an installed package. On an x86_64 server it
uses Docker Buildx/QEMU; runtime limits remain 3 CPUs and 4 GiB.

```sh
docker buildx build --platform linux/arm64 --load \
  --tag fast-ofm/core-validation:arm64 \
  --file validation/Dockerfile.arm64 .

docker run --rm --platform linux/arm64 --cpus 3 --memory 4g \
  fast-ofm/core-validation:arm64
```
