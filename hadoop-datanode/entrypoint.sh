#!/bin/bash

# Démarrer le serveur SSH
/usr/sbin/sshd

# Lancer le DataNode
echo "🚀 Démarrage du DataNode..."
exec hdfs datanode
