#!/bin/bash

# Démarrer le serveur SSH
/usr/sbin/sshd

# Format HDFS si ce n’est pas déjà fait
if [ ! -d "/hadoop/dfs/name/current" ]; then
    echo "🟢 Formatage du NameNode..."
    hdfs namenode -format -force
fi

# Lancer le NameNode
echo "🚀 Démarrage du NameNode..."
exec hdfs namenode
