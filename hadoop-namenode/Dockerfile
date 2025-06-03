FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

# Installation des outils nécessaires
RUN apt update && apt install -y \
    openjdk-8-jdk \
    openssh-server \
    rsync \
    wget \
    sudo \
    nano \
    net-tools \
    curl \
    sshpass \
    iputils-ping

# Variables d'environnement
ENV JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64
ENV HADOOP_VERSION=3.3.6
ENV HADOOP_HOME=/usr/local/hadoop
ENV PATH=$HADOOP_HOME/bin:$HADOOP_HOME/sbin:$JAVA_HOME/bin:$PATH

# Téléchargement et extraction de Hadoop
RUN wget https://downloads.apache.org/hadoop/common/hadoop-${HADOOP_VERSION}/hadoop-${HADOOP_VERSION}.tar.gz && \
    tar -xvzf hadoop-${HADOOP_VERSION}.tar.gz && \
    mv hadoop-${HADOOP_VERSION} /usr/local/hadoop && \
    rm hadoop-${HADOOP_VERSION}.tar.gz

# Création des dossiers pour le HDFS
RUN mkdir -p /hadoop/dfs/name

# Configuration SSH
RUN mkdir -p /run/sshd && \
    echo "root:root" | chpasswd && \
    sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config

# Copie des fichiers de config Hadoop
COPY core-site.xml $HADOOP_HOME/etc/hadoop/core-site.xml
COPY hdfs-site.xml $HADOOP_HOME/etc/hadoop/hdfs-site.xml

# Copie du script de démarrage
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 9870 9000 22

CMD ["/entrypoint.sh"]
