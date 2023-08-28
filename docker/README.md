# Gpu 버전으로 빌드하는 법

```bash
docker build -t faiss -f ./docker/Dockerfile.gpu ./docker/
```

In this command:
 - -t faiss specifies the name of the image.
 - -f ./docker/Dockerfile.gpu specifies the path to the Dockerfile.
 - ./docker/ at the end specifies the build context, which is the directory that contains resources used during the build.
