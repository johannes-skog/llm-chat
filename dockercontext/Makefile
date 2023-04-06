PROJECT_NAME=llama

clear:

	docker stop ${PROJECT_NAME}
	docker rm ${PROJECT_NAME}
	docker rmi ${PROJECT_NAME}

build:

	docker stop ${PROJECT_NAME} || true && docker rm ${PROJECT_NAME} || true
	docker build -t ${PROJECT_NAME} .
	docker run -dt -w /workspace  --name ${PROJECT_NAME} -v ${PWD}/../:/workspace -w /workspace -p 8081:8081 -p 8889:8889 -p 6667:6667 --env-file ../.env ${PROJECT_NAME} sleep infinity

enter:

	docker exec -it ${PROJECT_NAME}  /bin/bash

