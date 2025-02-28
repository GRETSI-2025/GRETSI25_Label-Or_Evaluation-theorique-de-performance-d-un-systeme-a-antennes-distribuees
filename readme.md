# recherche_reproductible_GRETSI25

Ce dépôt contient le code source associé à l’article *« Closed-Form BER for Distributed Antenna Systems in Partially Blocked Rayleigh Fading Channels »* présenté à la conférence GRETSI. L’objectif est de garantir la reproductibilité des résultats (label « recherche reproductible ») en fournissant l’ensemble des scripts et instructions nécessaires à la simulation et à l’analyse des performances BER en présence de blocages partiels.

## Table des matières

- [Installation](#installation)
- [Utilisation](#utilisation)
- [Configuration-et-Paramètres](#configuration-et-paramètres)
- [Reproductibilité-des-résultats](#reproductibilité-des-résultats)
- [Contribution](#contribution)
- [Licence](#licence)
- [Remerciements](#remerciements)

## Installation

Ce projet est développé sous Python 3 (nous recommandons Python 3.8 ou une version ultérieure).

### Création d’un environnement virtuel (venv)

Il est conseillé d’utiliser un environnement virtuel pour isoler les dépendances :

```bash
# Créer l’environnement virtuel
python3 -m venv env

# Lancer l’environnement (Linux/Mac)
source env/bin/activate

# Sous Windows
env\Scripts\activate

# Installation des dépendances
Les dépendances principales sont les suivantes :

TensorFlow (la version standard – CPU – est recommandée pour garantir la reproductibilité, même si vous pouvez utiliser un GPU si vous avez préalablement configuré vos pilotes et CUDA)
Sionna (librairie dédiée aux communications numériques)
D’autres librairies usuelles : NumPy, SciPy, Matplotlib, etc.
Pour installer l’ensemble des dépendances, exécutez :

bash
Copier
pip install -r requirements.txt
Exemple de contenu du fichier requirements.txt :

makefile
Copier
tensorflow==2.8.0           # ou la version recommandée
sionna==<version>           # à préciser (vérifiez la version utilisée dans votre projet)
numpy
scipy
matplotlib
Note :

Même si votre configuration dispose d’un GPU, la version standard de TensorFlow (non GPU) est indiquée pour garantir la reproductibilité sur machines CPU.
Si vous souhaitez exploiter le GPU, assurez-vous d’installer la version GPU de TensorFlow et de configurer correctement CUDA et cuDNN.
Utilisation
Pour reproduire les résultats présentés dans l’article, lancez simplement le script principal. Par exemple :

bash
Copier
python main.py
Le script principal exécute les simulations (par exemple, la génération des courbes BER) et utilise des paramètres définis au début du fichier. Ces paramètres incluent, entre autres :

Le nombre de BS (base stations)
La liste des clusters bloqués
Les puissances de symboles, le SNR, etc.
Remarque :
Si vous souhaitez voir la liste complète des paramètres et leur description, merci de joindre le code de la partie initiale (les premières lignes du fichier principal) afin de compléter cette section.

# Configuration-et-Paramètres
Les paramètres de simulation (nombre de BS, liste des clusters bloqués, SNR, etc.) sont définis en début de fichier dans le script principal. Vous pouvez les modifier directement dans le code pour adapter les simulations à vos besoins.

Exemple :

python
Copier
# Nombre de base stations
NUM_BS = 4

# Indice des clusters bloqués pour certaines BS
BLOCKED_CLUSTERS = {
    0: [1, 2, 3],  # Pour la BS 0, les clusters 1, 2 et 3 sont bloqués
    # ... autres configurations
}

# Paramètres de simulation
SNR_dB = 12  # Exemple de rapport Eb/N0 en dB
(N’hésitez pas à compléter ou ajuster cette section en fonction des détails de votre code.)

# Reproductibilité-des-résultats
Ce dépôt vise à assurer la reproductibilité de la recherche :

L’ensemble des scripts et configurations utilisés pour générer les résultats est fourni.
Les versions exactes des dépendances sont indiquées dans le fichier requirements.txt.
Des instructions détaillées (ci-dessus) expliquent comment configurer et exécuter le code, que ce soit sur une machine disposant d’un GPU ou uniquement d’un CPU.
Contribution
Les contributions sont les bienvenues. Pour signaler des bugs ou proposer des améliorations :

Ouvrez une issue dans ce dépôt.
Soumettez une pull request avec vos modifications.
Licence
(Précisez ici la licence applicable à votre projet, par exemple MIT, Apache, etc.)

# Remerciements
Ce travail a été financé en partie par le programme Horizon Europe (Hexa-X-II, grant No. 101095759).