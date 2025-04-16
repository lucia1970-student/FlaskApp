Cette application vous permetteras de déployer localement l'application flask vous permettant ainsi de télécharger un fichier .wav pour classifier le troubler du spectre autistique ainsi que visionner
et sauvegarder les charactéristiques de voix extrait à partir du fichier .wav télécharger.   Cette dernière fonctionnalité est implémenter avec la librarie praat-parselmouth.



Pour lancer l'application flask localement:

python3 ./app.py -r requirements.txt

NOTER: Les dépendances incluses dans le fichier requirements.txt devront être rencontrés.

1.  Les models sérialiser avec pickle sont inclus dans ../models et leurs implémentations propres sont incluses dans l'autre répertoire 'projet-finale'

2.  Le répertoire 'static' est conçu pour inclure les images, le javascript, le cascading style sheets, etc.

3.  Les fichiers dynamiques flask (html templates) sont contenus dans le répertoire ../templates.

4.  L'implémentation python des scripts praat ainsi que la pré validation et la conversion des fichiers .wav en format consistant sont incluse dans le répertoire ../utils

5.  Le fichier app.py est l'application 'main' de flask.
