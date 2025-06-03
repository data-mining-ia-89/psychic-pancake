#!/bin/bash
/usr/sbin/sshd

# Initialisation HDFS si non déjà formaté
if [ ! -d "/hadoop/dfs/name/current" ]; then
    echo "📦 Format du namenode..."
    hdfs namenode -format -force
fi

# Lancer le namenode
exec hdfs namenode
