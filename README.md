# Instalación
## Versión de Python
```
sys.version
'3.13.12 | packaged by Anaconda, Inc. | (main, Feb 24 2026, 16:13:31) [GCC 14.3.0]'
```
## Instalación de librerías
```
pip install -r requirements.txt
```
## Llamadas: 
Primero el train:
```
python ./plantilla_train.py config.json
```
En el train se generan los archivos preparados siempre con el mismo nombre por lo que en el test no hace falta ponerlos:
```
python ./plantilla_test.py
```