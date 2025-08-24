# recherche reproductible GRETSI25 : Évaluation théorique de performance d'un système à antennes  distribuées sur canaux multi-trajets partiellement bloqués

<hr>

**_Dépôt labelisé dans le cadre du [Label Reproductible du GRESTI'25](https://gretsi.fr/colloque2025/recherche-reproductible/)_**

| Label décerné | Auteur | Rapporteur | Éléments reproduits | Liens |
|:-------------:|:------:|:----------:|:-------------------:|:------|
| ![](label_or.png) | Thibaut ROLLAND<br>[@thibaut29](https://github.com/thibaut29) | Thomas MOREAU<br>[@tomMoral](https://github.com/tomMoral) |  Figures 1 à 3 | 📌&nbsp;[Dépôt&nbsp;original](https://gitlab.com/thibaut_29450/recherche_reproductible_gretsi25)<br>⚙️&nbsp;[Issue](https://github.com/GRETSI-2025/Label-Reproductible/issues/5)<br>📝&nbsp;[Rapport](https://github.com/akrah/test/tree/main/rapports/Rapport_issue_05) |

<hr>

Ce dépôt contient le code source associé à l’article *« Formule théorique de la probabilité d’erreur binaire d’un système MIMO distribué dans des canaux à multi-trajets partiellement bloqués »* présenté à la conférence GRETSI 2025 à Strasbourg. L’objectif est de garantir la reproductibilité des résultats (label « recherche reproductible ») en fournissant l’ensemble des scripts et instructions nécessaires à la simulation et à l’obtention des courbes de performances en terme de TEB présentées dans l'article.

## Table des matières

- [Installation](#installation)
- [Utilisation](#utilisation)
- [Configuration et Paramètres](#configuration-et-paramètres)
- [Reproductibilité des résultats](#reproductibilite-des-resultats)
- [Remerciements](#remerciements)
- [Licence](#licence)


## Installation

Ce projet est développé en Python et utilise la bibliothèque Sionna pour les simulations de communication numérique.

Version testée : sionna==1.0.2

Installation rapide :
```bash
pip install sionna==1.0.2
``` 
Pour plus de détails sur l’installation de Sionna, voir la documentation officielle :
[https://nvlabs.github.io/sionna/installation.html](https://nvlabs.github.io/sionna/installation.html)

### Prérequis

- **Python** : Assurez-vous d'avoir Python installé sur votre machine. Sionna nécessite Python 3.8 à 3.11.

- **TensorFlow** : Sionna requiert TensorFlow version 2.13 à 2.15.

- **Support GPU (optionnel)** : Si vous disposez d'un GPU compatible et souhaitez accélérer les simulations :

  - **CUDA Toolkit** : Installez le [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) correspondant à votre système.

  - **cuDNN** : Téléchargez et installez [cuDNN](https://developer.nvidia.com/cudnn) compatible avec votre version de CUDA.

  - **NVIDIA Container Toolkit** : Pour les utilisateurs de Docker souhaitant exploiter le GPU sous Linux, installez le [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

**Remarques :**

- Si vous n'avez pas de GPU, Sionna fonctionnera sur le CPU, bien que les simulations puissent être plus lentes.

- Sous macOS, il est nécessaire d'installer `tensorflow-macos` avant d'installer Sionna.

## Utilisation

Une fois l’environnement configuré, les scripts fournis permettent de générer les résultats de l’article et de visualiser les figures associées.

### Structure des fichiers

Le dépôt contient les fichiers suivants :

- **`main_Peb_and_MonteCarlo_simulation_DMIMO.py`** : Script principal permettant d’obtenir les résultats de TEB théorique et issus de simulations de Monte-Carlo en fonction des différents paramètres définis dans la section [Configuration et Paramètres](#configuration-et-paramètres).
- **`generation_figure1_et_figure2_illustration_PDF_CDF.py`** : Script générant les figures 1 et 2 illustrant les fonctions de densité de probabilité (PDF) et les fonctions de distribution cumulative (CDF) du signal reçu mal égalisé.
- **`generation_figure3_from_csv.py`** : Script générant la figure regroupant les valeurs obtenues pour les performances en TEB sous différentes conditions.
- **`useful_functions.py`** : Ce fichier contient diverses fonctions utilisées dans `main_Peb_and_MonteCarlo_simulation_DMIMO.py` pour la simulation et l’analyse des résultats.
- **`readme.md`** : Ce fichier contenant la documentation du dépôt.

### Exécution des simulations

#### 1. Calcul du TEB théorique et Monte-Carlo  
Exécutez le script principal pour obtenir les résultats de simulation :

```bash
python main_Peb_and_MonteCarlo_simulation_DMIMO.py
```
Le script va exécuter des simulations de Monte-Carlo pour estimer la probabilité d'erreur binaire (TEB) d’un système MIMO distribué dans des canaux à multi-trajets partiellement bloqués.
Les résultats (courbes TEB théorique et Monte-Carlo) sont automatiquement sauvegardés au format `.csv` dans le répertoire `results_csv/`.
##### Chargement des paramètres  
Le script commence par définir les paramètres du système, tels que :  
- Le **nombre de stations de base** (BS) et d'antennes utilisateur (UE)  
- Le **type de canal** CDL (ex. CDL-C, CDL-B) basé sur le modèle 3GPP TR38.901  
- Les **caractéristiques OFDM** : taille FFT, espacement des sous-porteuses, etc.  
- Le **schéma de modulation** utilisé (ex. QPSK avec 2 bits/symbole)  
- Le **niveau de blocage des clusters** dans chaque liaison BS–UE  
- Les **valeurs de $E_b/N_0$** testées pour évaluer le TEB  
- Le **nombre d’itérations Monte-Carlo** et d’erreurs cibles

Ces paramètres peuvent être ajustés dans le dictionnaire `SIMULATION_PARAM` au début du script `main_Peb_and_MonteCarlo_simulation_DMIMO.py`.

##### Exécution des simulations de Monte-Carlo  
- La simulation génère des signaux à partir de données binaires aléatoires transmis à travers le canal potentiellement bloqué.  
- Un **égaliseur** zero-forcing (ZF) est appliqué pour estimer le signal reçu.  
- Le signal estimé permet d'obtenir les symboles de la constellation puis, les bits associés sont comparés à ceux envoyés pour mesurer le **taux d'erreurs binaire**.  

##### Calcul de la probabilité d'erreur théorique   
La probabilité d'erreur binaire (TEB) théorique est calculée en utilisant les expressions analytiques dérivées dans l’article. Ces expressions prennent en compte :  

- Le **modèle de canal** non ligne de vue directe (NLoS) sélectionné (CDL-A, CDL-B et CDL-C),  
- La **présence de clusters bloqués**,  
- Le **rapport signal sur bruit (SNR)**,  
- L'impact de l'égalisation sur les performances du système.  

Les formules obtenues permettent d'estimer directement les performances du système **sans nécessiter de simulations Monte-Carlo**, ce qui est utile pour une validation rapide des résultats.  

Cette probabilité d'erreur est ensuite comparée aux résultats issus des simulations pour **vérifier la cohérence des modèles théoriques** avec les performances mesurées en conditions simulées. 

##### Affichage des résultats    
Le script génère principalement des graphiques montrant :  
- L’**évolution du TEB** en fonction du SNR.  
- Une **comparaison entre les résultats analytiques et les simulations de Monte-Carlo**.  

Les données numériques sont automatiquement sauvegardées sous forme de fichiers `.csv` dans le dossier `results_csv_test/`.

Pour afficher les courbes à partir de ces résultats, exécute le fichier suivant :  
```bash
python generation_figure3_from_csv.py
```
Cela ouvrira une figure comparant les TEBs théoriques et simulés pour chaque scénario de blocage.

### 2. Génération des figures PDF et CDF  

Pour afficher les **figures 1 et 2** de l'article :  
```bash
python generation_figure1_et_figure2_illustration_PDF_CDF.py
```

#### 3. Génération de la figure des performances en TEB  

Pour visualiser la **figure 3** regroupant les performances en TEB sous différentes conditions, exécutez la commande suivante :  

```bash
python generation_figure3.py
```

#### Personnalisation  
Si vous souhaitez personnaliser la simulation, vous pouvez **modifier les paramètres** dans le script avant de l’exécuter. 


## Configuration et Paramètres
Les paramètres de simulation sont définis en début de fichier dans le script principal. Vous pouvez les modifier directement dans le code pour adapter les simulations à vos besoins.

- **Nombre de symboles OFDM** : `num_ofdm_symbols = 14`  
- **Espacement des sous-porteuses** : `subcarrier_spacing = 15 kHz`  
- **Fréquence porteuse** : `carrier_frequency = 3.5 GHz`  
- **Taille de la FFT** : `fft_size = 102`  
- **Nombre de porteuses de garde** : `num_guard_carrier = [0, 0]`  
- **Nombre de bits par symbole** : `num_bits_per_symbol = 2`  
- **Puissance des stations de base (BS)** : `BS_power = 1/N_BS * np.ones(N_BS)`  
- **Nombre d’antennes utilisateur (UT)** : `num_ut_ant = 1`  
- **Nombre d’antennes par station de base (BS)** : `num_bs_ant = 1`  
- **Modèle de canal CDL (3GPP TR38.901)** : `cdl_model = ["C"] * N_BS`  
- **Étalement temporel du retard** : `delay_spread = 100 ns`  
- **Normalisation du canal** : `normalize_channel = False`  
- **Affichage des clusters** : `show_cluster = False`  
- **Modélisation des pertes de puissance sur les clusters** : `channel_blocked_equalizer = False`  
- **Nombre de clusters bloqués** : `nb_clusters_blocked = 0`  
- **Masque des clusters bloqués par lien** :  
 ` mask_clusters_blocked_per_link = [
      tf.constant([1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1], dtype=tf.complex64),
      tf.constant([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1], dtype=tf.complex64) `
  ]
  Cette variable est une **liste de tableaux** (`tf.constant`), où **chaque élément représente un masque de clusters bloqués pour une station de base spécifique**.  

  - **Le nombre d'éléments dans la liste doit être égal au nombre de stations de base (BS) sélectionnées.**  
  - **Chaque tableau `tf.constant` doit contenir autant de valeurs que de clusters dans le modèle de canal CDL sélectionné :**  

    - 📡 **23 valeurs** pour **CDL-A**  
    - 📡 **23 valeurs** pour **CDL-B**  
    - 📡 **24 valeurs** pour **CDL-C**  
    - 📡 **13 valeurs** pour **CDL-D**  
    - 📡 **14 valeurs** pour **CDL-E**  

  - Chaque valeur du tableau peut être :  
    - `1` (**cluster actif**)  
    - `0` (**cluster bloqué**)  

  Cela permet de **définir précisément les conditions de blocage** pour chaque station de base dans le scénario simulé.   
- **Mode de blocage DMIMO** : `DMIMO_blockage_mode = np.zeros([N_BS])`  
  - Définit l'état des liens entre les stations de base et l'utilisateur. Une valeur de `0` indique un lien bloqué.  

- **Type de précodage** : `precoding = ["steering_angles"]`  
  - Spécifie la technique de précodage utilisée. Options possibles : `"None"`, `"ZF"`.  

- **Mode de graphe TensorFlow** : `graph_mode = None`  
  - Permet d'activer ou non le mode graphe TensorFlow. Options possibles : `"None"`, `"xla"`, `"graph"`.  

- **Rapport signal sur bruit (SNR) en dB** :  

  ` ebno_db = list(np.arange(-30, 31, 2)) `
- **Taille de lot (`batch_size`)** : `2048`  
  - Ce paramètre définit le **nombre d'échantillons traités simultanément** dans la simulation.  
  - **Il dépend des ressources disponibles sur chaque PC** 💻.  
  - **⚠️ Attention :** Une valeur trop élevée peut entraîner une **erreur mémoire** ❌.  
  - **Recommandation** : Commencer avec une **valeur faible** et **augmenter progressivement** le nombre d'itérations (`max_mc_iter`) pour assurer une exécution stable.  

- **Nombre cible d’erreurs binaires (`num_target_bit_errors`)** : `None`  
  - Si défini, la simulation s'arrête une fois que ce nombre d'erreurs est atteint.  

- **BER cible (`target_ber`)** : `10e-6`  
  - Seuil de la **probabilité d'erreur binaire** recherché pour la simulation.  

- **Nombre maximal d’itérations Monte-Carlo (`max_mc_iter`)** : `10`  
  - Définit le **nombre d'itérations Monte-Carlo** pour garantir des résultats statistiquement significatifs.  
  - **Si `batch_size` est faible, il est conseillé d'augmenter `max_mc_iter` pour obtenir une meilleure précision**.  


## reproductibilite des resultats
Ce dépôt vise à assurer la reproductibilité de la recherche :
- L’ensemble des scripts et configurations utilisés pour générer les résultats est fourni.
- Des instructions détaillées (ci-dessus) expliquent comment configurer et exécuter le code, que ce soit sur une machine disposant d’un GPU ou uniquement d’un CPU.
- Les contributions sont les bienvenues. Pour signaler des bugs ou proposer des améliorations :
-- Ouvrez une issue dans ce dépôt.
-- Soumettez une pull request avec vos modifications.


## Remerciements
Ce travail a été financé en partie par le programme Horizon Europe (Hexa-X-II, grant No. 101095759).

## Licence et Citation

Ce projet utilise Sionna, une bibliothèque open-source sous licence Apache 2.0. Veuillez consulter le fichier [LICENSE](https://github.com/NVlabs/sionna/blob/main/LICENSE) pour plus de détails.

Si vous utilisez le logiciel Sionna dans vos travaux, merci de le citer comme suit :

```bibtex
@article{sionna,
    title = {Sionna: An Open-Source Library for Next-Generation Physical Layer Research},
    author = {Hoydis, Jakob and Cammerer, Sebastian and {Ait Aoudia}, Fayçal and Vem, Avinash and Binder, Nikolaus and Marcus, Guillermo and Keller, Alexander},
    year = {2022},
    month = {Mar.},
    journal = {arXiv preprint},
    online = {https://arxiv.org/abs/2203.11854}
}
```
