FROM ubuntu:20.04

# Éviter les invites interactives lors de l'installation
ENV DEBIAN_FRONTEND=noninteractive

# Installer les paquets nécessaires
RUN apt-get update && apt-get install -y \
    openjdk-8-jdk \
    openssh-server \
    net-tools \
    nano \
    wget \
    curl \
    sudo && \
    rm -rf /var/lib/apt/lists/*

# Créer l'utilisateur hadoop et l'ajouter à sudoers sans mot de passe
RUN useradd -m -s /bin/bash hadoop && \
    echo "hadoop ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Définir les variables d'environnement Java
ENV JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64
ENV PATH=$JAVA_HOME/bin:$PATH

# Télécharger et installer Hadoop
ENV HADOOP_VERSION=3.3.6
RUN wget https://downloads.apache.org/hadoop/common/hadoop-$HADOOP_VERSION/hadoop-$HADOOP_VERSION.tar.gz && \
    tar -xvzf hadoop-$HADOOP_VERSION.tar.gz && \
    mv hadoop-$HADOOP_VERSION /usr/local/hadoop && \
    rm hadoop-$HADOOP_VERSION.tar.gz

# Définir les variables d'environnement Hadoop
ENV HADOOP_HOME=/usr/local/hadoop
ENV PATH=$HADOOP_HOME/bin:$HADOOP_HOME/sbin:$PATH
ENV HDFS_NAMENODE_URI=hdfs://namenode:9000

# Mettre à jour le fichier hadoop-env.sh pour y inclure JAVA_HOME
RUN sed -i "s|^# export JAVA_HOME=.*|export JAVA_HOME=${JAVA_HOME}|" $HADOOP_HOME/etc/hadoop/hadoop-env.sh

# Changer la propriété du répertoire Hadoop pour l'utilisateur hadoop
RUN chown -R hadoop:hadoop /usr/local/hadoop

# Configurer SSH pour l'utilisateur hadoop
RUN mkdir -p /var/run/sshd && \
    mkdir -p /home/hadoop/.ssh && \
    chown -R hadoop:hadoop /home/hadoop/.ssh && \
    sudo -u hadoop ssh-keygen -t rsa -f /home/hadoop/.ssh/id_rsa -q -N "" && \
    cat /home/hadoop/.ssh/id_rsa.pub >> /home/hadoop/.ssh/authorized_keys && \
    chmod 600 /home/hadoop/.ssh/authorized_keys && \
    chmod 700 /home/hadoop/.ssh && \
    echo "StrictModes no" >> /etc/ssh/sshd_config && \
    echo "PasswordAuthentication yes" >> /etc/ssh/sshd_config

# Exposer le port SSH
EXPOSE 22

# Copier le script d'entrypoint
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Passer à l'utilisateur hadoop
USER hadoop
WORKDIR /home/hadoop

# Commande par défaut
CMD ["/bin/bash", "/entrypoint.sh", "-c", "tail -f /dev/null"]


