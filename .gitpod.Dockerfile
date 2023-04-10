# This will pull the official Gitpod `vnc` image
# which has much of what you need to start
FROM gitpod/workspace-full-vnc


# for ml specific files
USER 0
RUN mkdir -p "/etc/ml"

USER gitpod

COPY "./requirements.txt" "/etc/ml/ml_zero_requirements.txt"

# install the requirements
RUN pip3 install -r "/etc/ml/ml_zero_requirements.txt"

# run some command
